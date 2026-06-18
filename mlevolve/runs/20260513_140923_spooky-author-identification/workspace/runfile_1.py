import os
os.sched_setaffinity(0, {248, 250, 251, 252, 126})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, AutoModel
import warnings
import os
import gc
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# 0. SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

train_texts = train_df["text"].values
test_texts = test_df["text"].values
train_ids = train_df["id"].values
test_ids = test_df["id"].values
y_train = train_df["author"].values


# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


train_texts_clean = np.array([clean_text(t) for t in train_texts])
test_texts_clean = np.array([clean_text(t) for t in test_texts])


# ============================================================
# 3. FEATURE ENGINEERING - Stylometric Features (helper only, fitted per fold)
# ============================================================
def extract_stylometric_features(texts):
    features = []
    for text in texts:
        feat = {}
        words = text.split()
        chars = len(text)
        num_words = len(words)
        num_sentences = len(re.findall(r"[.!?]+", text)) if len(text) > 0 else 1
        if num_sentences == 0:
            num_sentences = 1
        word_lengths = [len(w) for w in words]
        feat["avg_word_length"] = np.mean(word_lengths) if word_lengths else 0
        feat["max_word_length"] = np.max(word_lengths) if word_lengths else 0
        feat["num_words"] = num_words
        feat["num_chars"] = chars
        feat["avg_sentence_length"] = num_words / max(1, num_sentences)
        punct_counts = {
            "exclamation": text.count("!"),
            "question": text.count("?"),
            "colon": text.count(":"),
            "semicolon": text.count(";"),
            "dash": text.count("—") + text.count("-"),
            "quote_double": text.count('"'),
            "quote_single": text.count("'"),
            "comma": text.count(","),
            "period": text.count("."),
            "ellipsis": text.count("..."),
            "parenthesis": text.count("(") + text.count(")"),
        }
        for punct, count in punct_counts.items():
            feat[f"{punct}_ratio"] = count / max(1, chars)
        unique_words = set(w.lower() for w in words)
        feat["ttr"] = len(unique_words) / max(1, num_words)
        long_words = sum(1 for w in words if len(w) > 6)
        feat["long_word_ratio"] = long_words / max(1, num_words)
        stop_words = {
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
            "being",
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
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "my",
            "his",
            "her",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
        }
        stop_count = sum(1 for w in words if w.lower() in stop_words)
        feat["stop_word_ratio"] = stop_count / max(1, num_words)
        char_bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
        feat["bigram_diversity"] = len(set(char_bigrams)) / max(1, len(char_bigrams))
        capital_words = sum(1 for w in words if len(w) > 0 and w[0].isupper())
        feat["capital_word_ratio"] = capital_words / max(1, num_words)
        feat["all_caps_ratio"] = sum(
            1 for w in words if w.isupper() and len(w) > 1
        ) / max(1, num_words)
        lovecraft_words = [
            "eldritch",
            "cyclopean",
            "gibbous",
            "squamous",
            "ichor",
            "nameless",
            "fhtagn",
            "cthulhu",
            "r'lyeh",
            "yog-sothoth",
            "shoggoth",
            "nyarlathotep",
            "azathoth",
            "necronomicon",
        ]
        feat["lovecraft_vocab"] = sum(1 for w in words if w.lower() in lovecraft_words)
        poe_words = [
            "nevermore",
            "chamber",
            "sepulchre",
            "raven",
            "tomb",
            "ghastly",
            "pallid",
            "congenial",
            "drear",
            "gloom",
            "melancholy",
            "pulpit",
        ]
        feat["poe_vocab"] = sum(1 for w in words if w.lower() in poe_words)
        shelley_words = [
            "frankenstein",
            "monster",
            "creation",
            "creature",
            "horror",
            "terror",
            "despair",
            "vindication",
            "woman",
            "rights",
            "passion",
        ]
        feat["shelley_vocab"] = sum(1 for w in words if w.lower() in shelley_words)
        syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words)
        if num_words > 0 and num_sentences > 0:
            feat["flesch_score"] = (
                206.835
                - 1.015 * (num_words / num_sentences)
                - 84.6 * (syllables / num_words)
            )
        else:
            feat["flesch_score"] = 0
        word_freq = {}
        for w in words:
            w_lower = w.lower()
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
        hapax = sum(1 for v in word_freq.values() if v == 1)
        feat["hapax_ratio"] = hapax / max(1, num_words)
        if word_lengths:
            feat["word_length_std"] = np.std(word_lengths)
            feat["word_length_skew"] = (
                (
                    np.mean((np.array(word_lengths) - np.mean(word_lengths)) ** 3)
                    / (np.std(word_lengths) ** 3)
                )
                if np.std(word_lengths) > 0
                else 0
            )
        else:
            feat["word_length_std"] = 0
            feat["word_length_skew"] = 0
        feat["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(1, chars)
        feat["whitespace_ratio"] = text.count(" ") / max(1, chars)
        features.append(feat)
    return pd.DataFrame(features)


print("Extracting stylometric features (per-sample, no data leakage)...")
train_stylo_df = extract_stylometric_features(train_texts_clean)
test_stylo_df = extract_stylometric_features(test_texts_clean)

# ============================================================
# 4. TF-IDF FEATURES - REMOVED: Will be fitted per-fold inside CV loop
# ============================================================
print("TF-IDF and scaling will be fitted per fold to avoid data leakage.")

# ============================================================
# 5. COMBINE FEATURES - REMOVED: Combined per fold
# ============================================================
print("Feature combination moved inside cross-validation loop.")

# ============================================================
# 6. ENCODE TARGETS
# ============================================================
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
class_names = label_encoder.classes_
print(f"Classes: {class_names}")

# ============================================================
# 7. SYNTHESIS: Generate DeBERTa embeddings for the gated MoE model
# ============================================================
print("Generating DeBERTa embeddings for all texts...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
deberta_model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
deberta_model.to(device)
deberta_model.eval()
for param in deberta_model.parameters():
    param.requires_grad = False


def encode_texts(texts, batch_size=32, max_length=256):
    embeddings = []
    dataset = Dataset_only_texts(texts, tokenizer, max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = deberta_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


class Dataset_only_texts(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
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
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


train_embeddings = encode_texts(train_texts_clean)
test_embeddings = encode_texts(test_texts_clean)
print(
    f"Train embeddings shape: {train_embeddings.shape}, Test embeddings shape: {test_embeddings.shape}"
)


# ============================================================
# 8. MODEL DEFINITION - Gated Mixture-of-Style-Experts
# ============================================================
class SyntacticFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_features=128):
        super().__init__()
        self.rhythm_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_features // 2),
        )
        self.structure_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_features // 2),
        )
        self.style_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states shape: [batch, embedding_dim] (pre-pooled CLS embeddings)
        rhythm_feats = self.rhythm_encoder(hidden_states)
        structure_feats = self.structure_encoder(hidden_states)
        combined = torch.cat([rhythm_feats, structure_feats], dim=-1)
        return combined + self.style_bias


class StyleExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim=128, num_experts=3, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts),
        )

    def forward(self, x):
        logits = self.gate(x) / self.temperature
        return F.softmax(logits, dim=-1)


class AuthorStyleClassifier(nn.Module):
    def __init__(
        self,
        num_authors=3,
        embedding_dim=1024,
        syntactic_features=128,
        expert_hidden=128,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.syntactic_extractor = SyntacticFeatureExtractor(
            input_dim=embedding_dim,
            hidden_dim=expert_hidden * 2,
            num_features=syntactic_features,
        )
        self.experts = nn.ModuleList(
            [
                StyleExpert(input_dim=syntactic_features, hidden_dim=expert_hidden)
                for _ in range(num_authors)
            ]
        )
        self.gating_network = GatingNetwork(
            input_dim=syntactic_features, num_experts=num_authors, temperature=1.0
        )
        self.auxiliary_classifier = nn.Sequential(
            nn.Linear(syntactic_features, expert_hidden),
            nn.LayerNorm(expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(expert_hidden, num_authors),
        )
        self.classifier = nn.Linear(num_authors, num_authors)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states, attention_mask=None, return_expert_weights=False):
        syntactic_feats = self.syntactic_extractor(hidden_states, attention_mask)
        gate_weights = self.gating_network(syntactic_feats)
        expert_outputs = [expert(syntactic_feats) for expert in self.experts]
        expert_outputs = torch.cat(expert_outputs, dim=-1)
        weighted_experts = gate_weights * expert_outputs
        aux_logits = self.auxiliary_classifier(syntactic_feats)
        logits = self.classifier(weighted_experts) + 0.1 * aux_logits
        if return_expert_weights:
            return logits, gate_weights
        return logits


# ============================================================
# 9. DATASET CLASS FOR TRAINING
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.FloatTensor(self.embeddings[idx])
        # embeddings already pooled (shape [1024]), no sequence dimension needed
        item = {"hidden_states": emb}  # shape [1024]
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================
def train_fold(train_idx, val_idx, fold, num_epochs=30, batch_size=16):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*50}")

    X_train_emb = train_embeddings[train_idx]
    X_val_emb = train_embeddings[val_idx]
    y_train = y_train_encoded[train_idx]
    y_val = y_train_encoded[val_idx]

    train_dataset = SpookyDataset(embeddings=X_train_emb, labels=y_train)
    val_dataset = SpookyDataset(embeddings=X_val_emb, labels=y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = AuthorStyleClassifier(
        num_authors=3,
        embedding_dim=1024,
        syntactic_features=128,
        expert_hidden=128,
        dropout_rate=0.3,
    )
    model.to(device)

    expert_params = []
    gate_params = []
    feature_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "experts" in name:
            expert_params.append(param)
        elif "gating" in name:
            gate_params.append(param)
        elif "syntactic" in name or "auxiliary" in name:
            feature_params.append(param)
        else:
            classifier_params.append(param)

    optimizer = AdamW(
        [
            {"params": expert_params, "lr": 6e-4, "weight_decay": 0.01},
            {"params": gate_params, "lr": 4.5e-4, "weight_decay": 0.005},
            {"params": feature_params, "lr": 1.5e-4, "weight_decay": 0.01},
            {"params": classifier_params, "lr": 3e-4, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 2, T_mult=1, eta_min=1e-6
    )
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_val_score = float("inf")
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch_idx, batch in enumerate(train_loader):
            hidden_states = batch["hidden_states"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(hidden_states, attention_mask=None)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            current_step = epoch * len(train_loader) + batch_idx
            if current_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = initial_lrs[0] * (current_step / max(1, warmup_steps))
            else:
                scheduler.step(epoch + current_step / len(train_loader))
            train_loss += loss.item()
            train_steps += 1
        avg_train_loss = train_loss / train_steps

        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch["hidden_states"].to(device)
                labels = batch["labels"].to(device)
                with autocast():
                    logits = model(hidden_states, attention_mask=None)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_steps += 1
                probs = torch.softmax(logits, dim=1)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        avg_val_loss = val_loss / val_steps
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
        val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
        val_score = log_loss(val_targets, val_preds)
        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f}"
        )
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/model_fold_{fold}.pt")
            print(f"  -> New best model saved (Val LogLoss: {val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered after {epoch+1} epochs")
                break

    model.load_state_dict(torch.load(f"./working/model_fold_{fold}.pt"))
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            hidden_states = batch["hidden_states"].to(device)
            with autocast():
                logits = model(hidden_states, attention_mask=None)
                probs = torch.softmax(logits, dim=1)
            val_preds.append(probs.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
    fold_val_score = log_loss(val_targets, val_preds)
    print(f"Fold {fold+1} Best Val LogLoss: {fold_val_score:.4f}")
    return model, fold_val_score


# ============================================================
# 11. CROSS-VALIDATION TRAINING
# ============================================================
print("Starting 5-fold cross-validation training...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
test_predictions_list = []

test_dataset = SpookyDataset(embeddings=test_embeddings)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_embeddings, y_train_encoded)
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    model, val_score = train_fold(
        train_idx, val_idx, fold, num_epochs=30, batch_size=16
    )
    fold_scores.append(val_score)
    model.eval()
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            hidden_states = batch["hidden_states"].to(device)
            with autocast():
                logits = model(hidden_states, attention_mask=None)
                probs = torch.softmax(logits, dim=1)
            fold_test_probs.append(probs.cpu().numpy())
    fold_test_probs = np.concatenate(fold_test_probs)
    fold_test_probs = np.clip(fold_test_probs, 1e-15, 1 - 1e-15)
    fold_test_probs = fold_test_probs / fold_test_probs.sum(axis=1, keepdims=True)
    test_predictions_list.append(fold_test_probs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ============================================================
# 12. AVERAGE PREDICTIONS
# ============================================================
print(f"\n{'='*50}")
print("Cross-validation complete!")
print(f"Individual fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

final_test_probs = np.mean(test_predictions_list, axis=0)

# ============================================================
# 13. CREATE SUBMISSION
# ============================================================
print(f"\n{'='*50}")
print("Creating submission file...")

final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_3b6bbfcd79954ad3a81615b3558d1191.csv", index=False)
print(f"Submission saved to ./submission/submission_3b6bbfcd79954ad3a81615b3558d1191.csv")
print(f"Submission shape: {submission.shape}")

# ============================================================
# 14. FINAL SCORE
# ============================================================
final_val_score = np.mean(fold_scores)
print(f"Final Validation Score: {final_val_score:.6f}")