import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.cuda.amp import autocast, GradScaler
import re
import os
import gc
import pickle

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Encode authors
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_authors = len(label_encoder.classes_)
print(f"Authors: {label_encoder.classes_}")

# Stratified split (80/20)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_df["text"], train_df["author_encoded"]))[0]

train_texts = train_df.iloc[train_idx]["text"].values
val_texts = train_df.iloc[val_idx]["text"].values
train_labels = train_df.iloc[train_idx]["author_encoded"].values
val_labels = train_df.iloc[val_idx]["author_encoded"].values
test_texts = test_df["text"].values

print(
    f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}"
)


# Clean text function
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# Apply cleaning
train_texts_clean = [clean_text(t) for t in train_texts]
val_texts_clean = [clean_text(t) for t in val_texts]
test_texts_clean = [clean_text(t) for t in test_texts]

# Feature Engineering: Character n-grams
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=1000,
    sublinear_tf=True,
    max_df=0.95,
    min_df=5,
)
char_features_train = char_vectorizer.fit_transform(train_texts_clean)
char_features_val = char_vectorizer.transform(val_texts_clean)
char_features_test = char_vectorizer.transform(test_texts_clean)
print(f"Character n-gram features: {char_features_train.shape[1]}")

# Feature Engineering: Word n-grams with TF-IDF
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=2000,
    sublinear_tf=True,
    max_df=0.85,
    min_df=5,
    stop_words="english",
)
word_features_train = word_vectorizer.fit_transform(train_texts_clean)
word_features_val = word_vectorizer.transform(val_texts_clean)
word_features_test = word_vectorizer.transform(test_texts_clean)
print(f"Word n-gram features: {word_features_train.shape[1]}")


# Feature Engineering: Punctuation patterns
def get_punct_features(text):
    features = []
    for punct in [".", "!", "?", ",", ";", ":", "-", '"', "'", "(", ")", "..."]:
        features.append(text.count(punct))
    punct_count = sum(1 for c in text if c in ".,!?;:'\"-")
    if len(text) > 0:
        features.append(punct_count / len(text))
    else:
        features.append(0)
    cap_count = sum(1 for c in text if c.isupper())
    if len(text) > 0:
        features.append(cap_count / len(text))
    else:
        features.append(0)
    words = text.split()
    if len(words) > 0:
        features.append(np.mean([len(w) for w in words]))
    else:
        features.append(0)
    return features


punct_train = np.array([get_punct_features(t) for t in train_texts_clean])
punct_val = np.array([get_punct_features(t) for t in val_texts_clean])
punct_test = np.array([get_punct_features(t) for t in test_texts_clean])
print(f"Punctuation features: {punct_train.shape[1]}")

# Combine all engineered features
X_train_engineered = np.hstack(
    [char_features_train.toarray(), word_features_train.toarray(), punct_train]
)
X_val_engineered = np.hstack(
    [char_features_val.toarray(), word_features_val.toarray(), punct_val]
)
X_test_engineered = np.hstack(
    [char_features_test.toarray(), word_features_test.toarray(), punct_test]
)
print(f"Engineered feature dimensions - Train: {X_train_engineered.shape}")

# Tokenize for DeBERTa
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts_clean,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
val_encodings = tokenizer(
    val_texts_clean, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
test_encodings = tokenizer(
    test_texts_clean, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
print(f"Tokenized train: {train_encodings['input_ids'].shape}")


# Model Architecture
class MultiScaleAuthorClassifier(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_authors=3,
        engineered_feat_dim=3043,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        transformer_dim = self.config.hidden_size
        self.stylo_encoder = nn.Sequential(
            nn.Linear(engineered_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=transformer_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.stylo_proj = nn.Linear(hidden_dim, transformer_dim)
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_authors),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [self.stylo_encoder, self.stylo_proj, self.classifier]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, stylo_features):
        transformer_outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_output = transformer_outputs.last_hidden_state[:, 0, :]
        stylo_out = self.stylo_encoder(stylo_features)
        stylo_projected = self.stylo_proj(stylo_out).unsqueeze(1)
        cls_unsqueezed = cls_output.unsqueeze(1)
        attn_output, _ = self.cross_attn(
            query=cls_unsqueezed, key=stylo_projected, value=stylo_projected
        )
        attn_output = attn_output.squeeze(1)
        fused_features = torch.cat([attn_output, stylo_out], dim=1)
        logits = self.classifier(fused_features)
        return logits


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engineered_feat_dim = X_train_engineered.shape[1]
model = MultiScaleAuthorClassifier(
    model_name="microsoft/deberta-v3-large",
    num_authors=num_authors,
    engineered_feat_dim=engineered_feat_dim,
    hidden_dim=256,
    dropout=0.3,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer with differential learning rates
transformer_params = [
    p for n, p in model.named_parameters() if "transformer" in n and p.requires_grad
]
stylo_params = [
    p
    for n, p in model.named_parameters()
    if "transformer" not in n and "classifier" not in n and p.requires_grad
]
classifier_params = [
    p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad
]

optimizer = torch.optim.AdamW(
    [
        {"params": transformer_params, "lr": 5e-6, "weight_decay": 0.01},
        {"params": stylo_params, "lr": 5e-4, "weight_decay": 0.01},
        {"params": classifier_params, "lr": 5e-4, "weight_decay": 0.005},
    ],
    eps=1e-8,
)

# Create DataLoaders
batch_size = 8
num_workers = 2

train_dataset = torch.utils.data.TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(X_train_engineered, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = torch.utils.data.TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(X_val_engineered, dtype=torch.float32),
    torch.tensor(val_labels, dtype=torch.long),
)
test_dataset = torch.utils.data.TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
    torch.tensor(X_test_engineered, dtype=torch.float32),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

# Training loop
num_epochs = 20
gradient_accumulation_steps = 2
scaler = GradScaler()
best_val_metric = float("inf")
patience = 5
patience_counter = 0
model_save_path = "./working/best_model.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, stylo_features, labels = [
            b.to(device) for b in batch
        ]
        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
            loss = criterion(logits, labels)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, stylo_features, labels = [
                b.to(device) for b in batch
            ]
            with autocast():
                logits = model(input_ids, attention_mask, stylo_features)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                probs = F.softmax(logits, dim=1)
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    eps = 1e-15
    # Check for and fix any NaN values in predictions
    if np.isnan(all_val_preds).any():
        all_val_preds = np.nan_to_num(all_val_preds, nan=1.0/num_authors)
    all_val_preds_clipped = np.clip(all_val_preds, eps, 1 - eps)
    val_metric = log_loss(all_val_labels, all_val_preds_clipped)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_metric:.4f}"
    )

    if val_metric < best_val_metric:
        best_val_metric = val_metric
        patience_counter = 0
        torch.save(
            {"model_state_dict": model.state_dict(), "val_metric": val_metric},
            model_save_path,
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model and compute final validation score
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_val_preds = []
all_val_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, stylo_features, labels = [
            b.to(device) for b in batch
        ]
        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
            probs = F.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

all_val_preds = np.concatenate(all_val_preds, axis=0)
all_val_labels = np.concatenate(all_val_labels, axis=0)
all_val_preds_clipped = np.clip(all_val_preds, eps, 1 - eps)
final_val_metric = log_loss(all_val_labels, all_val_preds_clipped)

# Test inference
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, stylo_features = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
            probs = F.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())

all_test_preds = np.concatenate(all_test_preds, axis=0)
row_sums = all_test_preds.sum(axis=1, keepdims=True)
all_test_preds_normalized = all_test_preds / row_sums

# Create submission
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "EAP": all_test_preds_normalized[:, 0],
        "HPL": all_test_preds_normalized[:, 1],
        "MWS": all_test_preds_normalized[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Final Validation Score: {final_val_metric}")
