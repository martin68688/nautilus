import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ───
NUM_AUTHORS = 3
MODEL_NAME = "microsoft/deberta-v3-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── Load data ───
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")


# ─── Basic text cleaning ───
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


train["text_clean"] = train["text"].apply(clean_text)
test["text_clean"] = test["text"].apply(clean_text)


# ─── Feature engineering ───
def extract_style_features(texts, fit_scaler=False, scaler=None):
    features_list = []
    for text in texts:
        if not text or len(text.strip()) == 0:
            features_list.append([0] * 30)
            continue
        char_count = len(text)
        word_count = len(text.split())
        sent_count = len(re.split(r"[.!?]+", text))
        avg_word_len = char_count / max(word_count, 1)
        avg_sent_len = word_count / max(sent_count, 1)
        exclamation_count = text.count("!")
        question_count = text.count("?")
        comma_count = text.count(",")
        semicolon_count = text.count(";")
        colon_count = text.count(":")
        dash_count = text.count("—") + text.count("–") + text.count("-")
        quote_count = text.count('"') + text.count('"') + text.count("'")
        paren_count = text.count("(") + text.count(")")
        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        title_case_words = sum(1 for w in text.split() if w.istitle())
        all_caps_ratio = caps_words / max(word_count, 1)
        title_case_ratio = title_case_words / max(word_count, 1)
        archaic_words = sum(
            1
            for w in text.lower().split()
            if w
            in [
                "thou",
                "thee",
                "thy",
                "thine",
                "hath",
                "doth",
                "dost",
                "ere",
                "whence",
                "thence",
                "hither",
                "thither",
                "ye",
                "forsooth",
                "methinks",
                "perchance",
                "prithee",
                "wherefore",
                "thus",
            ]
        )
        contractions = sum(1 for w in text.lower().split() if "'" in w and len(w) > 2)
        words = text.lower().split()
        unique_words = len(set(words))
        type_token_ratio = unique_words / max(word_count, 1)
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / max(word_count, 1)
        stopwords = {
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
            "was",
            "were",
            "had",
            "has",
            "have",
            "been",
            "being",
            "is",
            "are",
            "be",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "then",
            "than",
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
            "all",
            "each",
            "every",
            "some",
            "any",
            "many",
            "much",
            "more",
            "most",
            "such",
            "only",
            "own",
            "same",
            "very",
            "just",
            "too",
            "also",
            "now",
            "then",
            "here",
            "there",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
        }
        stopword_count = sum(1 for w in words if w in stopwords)
        stopword_ratio = stopword_count / max(word_count, 1)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        starts_with_conjunction = sum(
            1
            for s in sentences
            if len(s.split()) > 0
            and s.split()[0].lower() in ["and", "but", "or", "nor", "yet", "so", "for"]
        )
        conjunction_start_ratio = starts_with_conjunction / max(len(sentences), 1)
        punct_count = sum(1 for c in text if c in ".,;:!?\"'()[]{}")
        punct_density = punct_count / max(char_count, 1)
        first_person = sum(
            1
            for w in words
            if w
            in [
                "i",
                "me",
                "my",
                "mine",
                "myself",
                "we",
                "us",
                "our",
                "ours",
                "ourselves",
            ]
        )
        first_person_ratio = first_person / max(word_count, 1)
        third_person = sum(
            1
            for w in words
            if w
            in [
                "he",
                "him",
                "his",
                "she",
                "her",
                "hers",
                "they",
                "them",
                "their",
                "theirs",
            ]
        )
        third_person_ratio = third_person / max(word_count, 1)
        features = [
            char_count,
            word_count,
            sent_count,
            avg_word_len,
            avg_sent_len,
            exclamation_count,
            question_count,
            comma_count,
            semicolon_count,
            colon_count,
            dash_count,
            quote_count,
            paren_count,
            all_caps_ratio,
            title_case_ratio,
            archaic_words,
            contractions,
            type_token_ratio,
            long_word_ratio,
            stopword_ratio,
            conjunction_start_ratio,
            punct_density,
            first_person_ratio,
            third_person_ratio,
        ]
        features_normalized = [
            exclamation_count / max(char_count, 1),
            question_count / max(char_count, 1),
            comma_count / max(char_count, 1),
            semicolon_count / max(char_count, 1),
            colon_count / max(char_count, 1),
            dash_count / max(char_count, 1),
        ]
        features.extend(features_normalized)
        features_list.append(features)
    features_array = np.array(features_list, dtype=np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        return scaler.fit_transform(features_array), scaler
    else:
        return (
            scaler.transform(features_array) if scaler is not None else features_array
        )


train_features, feature_scaler = extract_style_features(
    train["text_clean"].values, fit_scaler=True
)
test_features = extract_style_features(
    test["text_clean"].values, fit_scaler=False, scaler=feature_scaler
)
print(
    f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}"
)

# ─── Encode labels ───
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train["author"].values)
print(f"Classes: {label_encoder.classes_}")

# ─── Stratified split ───
train_texts, val_texts, train_feats, val_feats, train_labels_split, val_labels = (
    train_test_split(
        train["text_clean"].values,
        train_features,
        train_labels,
        test_size=0.15,
        random_state=42,
        stratify=train_labels,
    )
)
print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

# ─── Save processed data ───
np.save("./working/train_texts.npy", train_texts)
np.save("./working/val_texts.npy", val_texts)
np.save("./working/test_texts.npy", test["text_clean"].values)
np.save("./working/train_labels.npy", train_labels_split)
np.save("./working/val_labels.npy", val_labels)
np.save("./working/test_ids.npy", test["id"].values)
np.save("./working/full_train_texts.npy", train["text_clean"].values)
np.save("./working/full_train_labels.npy", train_labels)


# ─── Model Definition ───
class AuthorClassifier(nn.Module):
    def __init__(
        self,
        model_name=MODEL_NAME,
        num_labels=NUM_AUTHORS,
        hidden_size=512,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        backbone_hidden = self.config.hidden_size
        self.attention_pool = nn.Sequential(
            nn.Linear(backbone_hidden, backbone_hidden // 2),
            nn.Tanh(),
            nn.Linear(backbone_hidden // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_hidden),
            nn.Linear(backbone_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_size, num_labels),
        )
        self.residual_proj = nn.Linear(backbone_hidden, hidden_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in [self.classifier, self.attention_pool, self.residual_proj]:
            if hasattr(module, "weight") and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight, gain=0.5)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attn_scores = self.attention_pool(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        bottleneck = self.classifier[0:4](pooled)
        block1 = self.classifier[4:8](bottleneck)
        block1 = block1 + self.residual_proj(pooled) * 0.3
        block2 = self.classifier[8:12](block1)
        block2 = block2 + block1 * 0.3
        logits = self.classifier[12](block2)
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, targets):
        n_classes = preds.size(1)
        with torch.no_grad():
            smoothed_targets = torch.full_like(preds, self.epsilon / (n_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        log_probs = torch.log_softmax(preds, dim=1)
        loss = -(smoothed_targets * log_probs).sum(dim=1)
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum() if self.reduction == "sum" else loss
        )


# ─── Initialize model, tokenizer, optimizer ───
model = AuthorClassifier()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
model.to(device)

backbone_params = [
    p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
]
head_params = [
    p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
]
optimizer = AdamW(
    [
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-4, "weight_decay": 0.01},
    ],
    weight_decay=0.01,
)

criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
scaler = GradScaler()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# ─── Tokenization ───
print("Tokenizing texts...")
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)
test_encodings = tokenizer(
    test["text_clean"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)

train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels_split, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

# ─── Training ───
num_epochs = 20
patience = 3
best_val_loss = float("inf")
epochs_no_improve = 0
best_model_path = "./working/best_model.pt"
total_steps = len(train_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-5,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy="linear",
)

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            all_val_preds.append(probs.cpu().numpy())
            all_val_targets.append(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_targets = np.concatenate(all_val_targets, axis=0)
    eps = 1e-15
    val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )
    val_log_loss = -np.mean(
        np.sum(np.eye(3)[val_targets] * np.log(val_preds_normalized), axis=1)
    )
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_log_loss:.4f}"
    )
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ─── Load best model ───
model.load_state_dict(torch.load(best_model_path))
model.eval()

# ─── Final validation ───
all_val_preds = []
all_val_targets = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        all_val_preds.append(probs.cpu().numpy())
        all_val_targets.append(labels.cpu().numpy())
val_preds = np.concatenate(all_val_preds, axis=0)
val_targets = np.concatenate(all_val_targets, axis=0)
eps = 1e-15
val_preds_clipped = np.clip(val_preds, eps, 1 - eps)
val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
final_val_log_loss = -np.mean(
    np.sum(np.eye(3)[val_targets] * np.log(val_preds_normalized), axis=1)
)

# ─── Test inference ───
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        all_test_preds.append(probs.cpu().numpy())
test_preds = np.concatenate(all_test_preds, axis=0)

# ─── Save submission ───
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {
        "id": test["id"].values,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with shape {submission_df.shape}"
)
print(f"Final Validation Score: {final_val_log_loss:.6f}")
