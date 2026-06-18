import os
os.sched_setaffinity(0, {6, 19, 20, 21, 22, 23, 24, 25, 26, 27})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import warnings
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertModel
from transformers.optimization import get_cosine_schedule_with_warmup

# Download stopwords if needed
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])

# ===================== DATA PROCESSING & FEATURE ENGINEERING =====================

# DistilBERT tokenizer
print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_seq_len = 256

def compute_stylometric_features(text):
    """Compute a comprehensive set of 26 handcrafted stylometric features."""
    if not isinstance(text, str) or len(text) == 0:
        return np.zeros(26)

    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)

    # Word-level features
    # 1. Average word length
    avg_word_len = sum(len(w) for w in words) / max(num_words, 1)

    # 2. Log-transformed sentence length (in words)
    sentence_len_log = np.log1p(num_words / max(num_sentences, 1))

    # 3. Punctuation density (total)
    total_punct = sum(1 for c in text if c in '.,!?;:()[]{}""''')
    punct_density = total_punct / max(num_chars, 1)

    # 4. Stopword ratio
    tokens_lower = [w.lower() for w in words]
    stopword_count = sum(1 for t in tokens_lower if t in stop_words)
    stopword_ratio = stopword_count / max(num_words, 1)

    # 5. Unique word ratio (type-token ratio)
    unique_words = len(set(tokens_lower))
    unique_ratio = unique_words / max(num_words, 1)

    # 6. Coleman-Lieu readability index
    L = (num_chars / max(num_words, 1)) * 100
    S_val = (num_sentences / max(num_words, 1)) * 100
    coleman_lieu = 0.0588 * L - 0.296 * S_val - 15.8

    # 7. Average sentence length in characters
    avg_sent_len_chars = num_chars / max(num_sentences, 1)

    # 8. Number of sentences (log-transformed)
    num_sent_log = np.log1p(num_sentences)

    # 9. Ratio of long words (>6 characters)
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / max(num_words, 1)

    # 10. Ratio of short words (<3 characters)
    short_words = sum(1 for w in words if len(w) < 3)
    short_word_ratio = short_words / max(num_words, 1)

    # 11. Ratio of all-caps words
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    all_caps_ratio = all_caps / max(num_words, 1)

    # 12. Ratio of words starting with capital letter
    capitalized = sum(1 for w in words if w[0].isupper() if w)
    cap_ratio = capitalized / max(num_words, 1)

    # 13. Ratio of digits
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(num_chars, 1)

    # 14. Hapax legomena ratio (words appearing once)
    word_freq = Counter(tokens_lower)
    hapax_leg = sum(1 for v in word_freq.values() if v == 1)
    hapax_leg_ratio = hapax_leg / max(len(tokens_lower), 1)

    # 15. Hapax dislegomena ratio (words appearing twice)
    hapax_dis = sum(1 for v in word_freq.values() if v == 2)
    hapax_dis_ratio = hapax_dis / max(len(tokens_lower), 1)

    # 16. Yule's K (lexical richness measure)
    S1 = len(tokens_lower)
    S2 = sum(v * (v - 1) for v in word_freq.values())
    yules_K = 10000 * (S2 - S1) / max(S1 * S1, 1) if S1 > 0 else 0

    # 17. Sentence length standard deviation
    sent_lengths = [len(s.split()) for s in sentences]
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # 18. Paragraph count (split by double newline)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    para_count = len(paragraphs)

    # Punctuation-specific features
    # 19. Comma density
    comma_count = text.count(',')
    comma_density = comma_count / max(num_chars, 1)

    # 20. Exclamation density
    excl_count = text.count('!')
    excl_density = excl_count / max(num_chars, 1)

    # 21. Question density
    ques_count = text.count('?')
    ques_density = ques_count / max(num_chars, 1)

    # 22. Colon/semicolon density
    colon_count = text.count(':') + text.count(';')
    colon_density = colon_count / max(num_chars, 1)

    # 23. Quote density
    quote_count = text.count('"') + text.count("'")
    quote_density = quote_count / max(num_chars, 1)

    # 24. Dash density
    dash_count = text.count('-') + text.count('—') + text.count('–')
    dash_density = dash_count / max(num_chars, 1)

    # 25. Ellipsis density
    ellipsis_count = text.count('...')
    ellipsis_density = ellipsis_count / max(num_chars, 1)

    # 26. Bracket/parenthesis density
    bracket_count = text.count('(') + text.count(')') + text.count('[') + text.count(']') + text.count('{') + text.count('}')
    bracket_density = bracket_count / max(num_chars, 1)

    return np.array([
        avg_word_len, sentence_len_log, punct_density, stopword_ratio, unique_ratio,
        coleman_lieu, avg_sent_len_chars, num_sent_log, long_word_ratio, short_word_ratio,
        all_caps_ratio, cap_ratio, digit_ratio, hapax_leg_ratio, hapax_dis_ratio,
        yules_K, sent_len_std, para_count,
        comma_density, excl_density, ques_density, colon_density,
        quote_density, dash_density, ellipsis_density, bracket_density
    ])

# Split BEFORE tokenization
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["author_encoded"],
)

train_df_train = train_df.iloc[train_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train split size: {len(train_df_train)}, Val split size: {len(train_df_val)}")

# Tokenize with DistilBERT tokenizer
print("Tokenizing with DistilBERT...")
def tokenize_texts(texts, tokenizer, max_len):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, 'tolist') else list(texts),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

X_train_ids, X_train_mask = tokenize_texts(train_df_train["text"], tokenizer, max_seq_len)
X_val_ids, X_val_mask = tokenize_texts(train_df_val["text"], tokenizer, max_seq_len)
X_test_ids, X_test_mask = tokenize_texts(test_df["text"], tokenizer, max_seq_len)

y_train = train_df_train["author_encoded"].values
y_val = train_df_val["author_encoded"].values

print(f"Train IDs: {X_train_ids.shape}, Val IDs: {X_val_ids.shape}, Test IDs: {X_test_ids.shape}")

# Compute stylometric features
print("Computing stylometric features...")
stylo_features_train = np.array([compute_stylometric_features(text) for text in train_df_train["text"]])
stylo_features_val = np.array([compute_stylometric_features(text) for text in train_df_val["text"]])
stylo_features_test = np.array([compute_stylometric_features(text) for text in test_df["text"]])

# Normalize stylometric features
scaler = StandardScaler()
stylo_features_train = scaler.fit_transform(stylo_features_train)
stylo_features_val = scaler.transform(stylo_features_val)
stylo_features_test = scaler.transform(stylo_features_test)

print(f"Stylometric features shape - Train: {stylo_features_train.shape}, Val: {stylo_features_val.shape}, Test: {stylo_features_test.shape}")

# ===================== MODEL DESIGN =====================

class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes=3, stylo_dim=26, dropout=0.2):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze BERT initially
        for param in self.bert.parameters():
            param.requires_grad = False

        # Classification head for BERT output (768-dim)
        self.bert_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

        # Stylometric MLP
        self.stylo_mlp = nn.Sequential(
            nn.Linear(stylo_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )

        # Final fusion layer
        self.fusion = nn.Linear(3 + 32, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, stylo_features=None):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        # BERT classification head
        bert_out = self.bert_classifier(pooled)  # (batch, 3)

        if stylo_features is not None:
            stylo_out = self.stylo_mlp(stylo_features)  # (batch, 32)
            combined = torch.cat([bert_out, stylo_out], dim=1)  # (batch, 35)
        else:
            combined = bert_out

        combined = self.dropout(combined)
        logits = self.fusion(combined)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DistilBERTClassifier(
    num_classes=3,
    stylo_dim=26,
    dropout=0.2
).to(device)

# Loss function with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_pred = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_pred, dim=1))

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Separate parameter groups for different learning rates
bert_params = []
head_params = []
for name, param in model.named_parameters():
    if 'bert' in name:
        bert_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': bert_params, 'lr': 2e-5},
    {'params': head_params, 'lr': 1e-3}
], weight_decay=1e-4)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"BERT params: {sum(p.numel() for p in bert_params):,}, Head params: {sum(p.numel() for p in head_params):,}")

# ===================== TRAINING & EVALUATION =====================

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, stylo_features, labels=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.stylo_features = torch.FloatTensor(stylo_features)
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.input_ids[idx], self.attention_masks[idx], self.stylo_features[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_masks[idx], self.stylo_features[idx]

# Create datasets and dataloaders
train_dataset = TextDataset(X_train_ids, X_train_mask, stylo_features_train, y_train)
val_dataset = TextDataset(X_val_ids, X_val_mask, stylo_features_val, y_val)
test_dataset = TextDataset(X_test_ids, X_test_mask, stylo_features_test)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

# Exponential Moving Average
num_epochs = 8
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Phase 1: Train only head (freeze BERT)
print("Phase 1: Training only classification head (BERT frozen)...")
for epoch in range(3):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_ids, batch_mask, batch_stylo, batch_y in train_loader:
        batch_ids, batch_mask, batch_stylo, batch_y = batch_ids.to(device), batch_mask.to(device), batch_stylo.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_ids, batch_mask, stylo_features=batch_stylo)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * batch_ids.size(0)
        _, predicted = torch.max(logits, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_labels_list = []
    with torch.no_grad():
        for batch_ids, batch_mask, batch_stylo, batch_y in val_loader:
            batch_ids, batch_mask, batch_stylo, batch_y = batch_ids.to(device), batch_mask.to(device), batch_stylo.to(device), batch_y.to(device)
            logits = model(batch_ids, batch_mask, stylo_features=batch_stylo)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_ids.size(0)
            probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_labels_list.append(batch_y.cpu().numpy())
            _, predicted = torch.max(logits, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels_list)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_log_loss = log_loss(val_labels, val_probs)

    print(f"Phase1 Epoch {epoch+1:2d}/3 | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} LogLoss: {val_log_loss:.4f}")

# Phase 2: Unfreeze BERT and fine-tune all
print("Phase 2: Fine-tuning all parameters...")
for param in model.bert.parameters():
    param.requires_grad = True

# Collect parameters again after unfreezing
bert_unfrozen_params = []
head_unfrozen_params = []
for name, param in model.named_parameters():
    if 'bert' in name:
        bert_unfrozen_params.append(param)
    else:
        head_unfrozen_params.append(param)

# Re-create optimizer with all parameters unfrozen and lower BERT LR
optimizer = torch.optim.AdamW([
    {'params': bert_unfrozen_params, 'lr': 2e-5},
    {'params': head_unfrozen_params, 'lr': 1e-3}
], weight_decay=1e-4)

# Re-create scheduler for remaining steps
remaining_steps = num_epochs * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * remaining_steps),
    num_training_steps=remaining_steps
)

best_val_loss = float("inf")
best_model_state = None
patience = 5
patience_counter = 0

# Enable gradient checkpointing to reduce memory
model.bert.gradient_checkpointing_enable()

print("Starting fine-tuning...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_ids, batch_mask, batch_stylo, batch_y in train_loader:
        batch_ids, batch_mask, batch_stylo, batch_y = batch_ids.to(device), batch_mask.to(device), batch_stylo.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_ids, batch_mask, stylo_features=batch_stylo)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * batch_ids.size(0)
        _, predicted = torch.max(logits, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_probs = []
    val_labels_list = []

    with torch.no_grad():
        for batch_ids, batch_mask, batch_stylo, batch_y in val_loader:
            batch_ids, batch_mask, batch_stylo, batch_y = batch_ids.to(device), batch_mask.to(device), batch_stylo.to(device), batch_y.to(device)
            logits = model(batch_ids, batch_mask, stylo_features=batch_stylo)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * batch_ids.size(0)
            probs = torch.softmax(logits, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_labels_list.append(batch_y.cpu().numpy())
            _, predicted = torch.max(logits, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels_list)
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_log_loss = log_loss(val_labels, val_probs)

    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} LogLoss: {val_log_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "./working/best_model_1d2df3e0123340b4b9867f88affddc2f.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model and compute final validation score
model.load_state_dict(best_model_state)
model.eval()
val_probs = []
with torch.no_grad():
    for batch_ids, batch_mask, batch_stylo, _ in val_loader:
        batch_ids, batch_mask, batch_stylo = batch_ids.to(device), batch_mask.to(device), batch_stylo.to(device)
        logits = model(batch_ids, batch_mask, stylo_features=batch_stylo)
        probs = torch.softmax(logits, dim=1)
        val_probs.append(probs.cpu().numpy())

val_probs = np.concatenate(val_probs)
val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)
final_val_score = log_loss(y_val, val_probs)

print(f"Final Validation Score: {final_val_score}")

# Generate test predictions
print("Generating test predictions...")
model.eval()
test_probs = []
with torch.no_grad():
    for batch_ids, batch_mask, batch_stylo in test_loader:
        batch_ids, batch_mask, batch_stylo = batch_ids.to(device), batch_mask.to(device), batch_stylo.to(device)
        logits = model(batch_ids, batch_mask, stylo_features=batch_stylo)
        probs = torch.softmax(logits, dim=1)
        test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs)
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# Create submission file
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_1d2df3e0123340b4b9867f88affddc2f.csv", index=False)
print(f"Submission saved to ./submission/submission_1d2df3e0123340b4b9867f88affddc2f.csv")
print(f"Submission shape: {submission.shape}")