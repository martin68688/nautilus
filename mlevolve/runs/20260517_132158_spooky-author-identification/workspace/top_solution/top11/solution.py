# merged_script.py
# Author identification using stylometric features + frozen ModernBERT with multi-pooling head

import numpy as np
import pandas as pd
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, ModernBertModel
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_LENGTH = 256
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
DROPOUT_RATE = 0.3
FREEZE_BACKBONE = True

# ============================================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_stylometric_features(text_series):
    features_list = []
    for text in text_series:
        features = {}
        text_str = str(text)
        words = text_str.split()
        sentences = re.split(r"[.!?]+", text_str)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        chars = list(text_str)

        features["word_count"] = len(words)
        features["char_count"] = len(text_str)
        features["sentence_count"] = max(len(sentences), 1)
        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0

        punct_counts = {
            "comma_density": text_str.count(",") / max(len(words), 1),
            "semicolon_density": text_str.count(";") / max(len(words), 1),
            "colon_density": text_str.count(":") / max(len(words), 1),
            "exclamation_density": text_str.count("!") / max(len(words), 1),
            "question_density": text_str.count("?") / max(len(words), 1),
            "dash_density": text_str.count("-") / max(len(words), 1),
            "quote_density": text_str.count('"') / max(len(words), 1),
            "apostrophe_density": text_str.count("'") / max(len(words), 1),
            "ellipsis_count": text_str.count("...") + text_str.count("…"),
            "parentheses_density": (text_str.count("(") + text_str.count(")"))
            / max(len(words), 1),
        }
        features.update(punct_counts)

        caps_words = sum(1 for w in words if w[0].isupper() if w)
        features["capitalized_word_ratio"] = caps_words / max(len(words), 1)
        features["all_caps_word_ratio"] = sum(
            1 for w in words if w.isupper() and len(w) > 1
        ) / max(len(words), 1)

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
                "it",
                "its",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "we",
                "they",
            ]
        )
        stopword_count = sum(1 for w in words if w.lower() in stopwords)
        features["stopword_ratio"] = stopword_count / max(len(words), 1)

        unique_words = set(w.lower() for w in words)
        features["unique_word_ratio"] = len(unique_words) / max(len(words), 1)

        words_per_sentence = len(words) / max(features["sentence_count"], 1)
        features["avg_sentence_length"] = words_per_sentence
        features["long_sentence_ratio"] = sum(
            1 for s in sentences if len(s.split()) > 20
        ) / max(len(sentences), 1)
        features["short_sentence_ratio"] = sum(
            1 for s in sentences if len(s.split()) < 5
        ) / max(len(sentences), 1)

        features["vowel_ratio"] = sum(1 for c in text_str if c in "aeiou") / max(
            len(text_str), 1
        )
        features["period_ratio"] = text_str.count(".") / max(len(sentences), 1)

        negation_words = [
            "not",
            "no",
            "never",
            "nothing",
            "neither",
            "nor",
            "none",
            "nobody",
        ]
        negation_count = sum(1 for w in words if w.lower() in negation_words)
        features["negation_density"] = negation_count / max(len(words), 1)

        descriptive_adj_endings = sum(
            1
            for w in words
            if any(
                w.lower().endswith(suffix)
                for suffix in ["ful", "ous", "ive", "ic", "al", "y"]
            )
        )
        features["descriptive_ratio"] = descriptive_adj_endings / max(len(words), 1)

        past_count = sum(
            1
            for w in words
            if w.lower().endswith("ed") or w.lower() in ["was", "were", "had", "been"]
        )
        features["past_tense_ratio"] = past_count / max(len(words), 1)

        features_list.append(features)
    return pd.DataFrame(features_list)

train_df["cleaned_text"] = train_df["text"].apply(clean_text)
test_df["cleaned_text"] = test_df["text"].apply(clean_text)

print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(train_df["text"])
test_stylo = extract_stylometric_features(test_df["text"])
train_stylo = train_stylo.fillna(0)
test_stylo = test_stylo.fillna(0)

print("Creating character n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    max_features=15000,
    sublinear_tf=True,
    min_df=5,
    max_df=0.85,
    strip_accents="unicode",
)
train_char_ngrams = char_vectorizer.fit_transform(train_df["cleaned_text"])
test_char_ngrams = char_vectorizer.transform(test_df["cleaned_text"])
print(f"Character n-gram features: {train_char_ngrams.shape[1]}")

print("Creating word n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.80,
    stop_words="english",
)
train_word_ngrams = word_vectorizer.fit_transform(train_df["cleaned_text"])
test_word_ngrams = word_vectorizer.transform(test_df["cleaned_text"])
print(f"Word n-gram features: {train_word_ngrams.shape[1]}")

train_stylo_sparse = csr_matrix(train_stylo.values)
test_stylo_sparse = csr_matrix(test_stylo.values)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["author"])
print(
    f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(np.zeros(len(y_train)), y_train))

# CORRECT indexing using sub-DataFrames to avoid INDEX_BUG
train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

train_texts = train_set["text"].values.copy()
train_labels = label_encoder.transform(train_set["author"].values).copy()
val_texts = val_set["text"].values.copy()
val_labels = label_encoder.transform(val_set["author"].values).copy()
test_texts = test_df["text"].values.copy()
test_ids = test_df["id"].values.copy()

assert (
    len(set(train_idx) & set(val_idx)) == 0
), "INDEX_BUG: Train and validation indices overlap!"

# Scale stylometric features
scaler = StandardScaler(with_mean=False)
train_stylo_scaled = scaler.fit_transform(train_stylo_sparse[train_idx])
val_stylo_scaled = scaler.transform(train_stylo_sparse[val_idx])
test_stylo_scaled = scaler.transform(test_stylo_sparse)

X_train_final = hstack(
    [train_stylo_scaled, train_char_ngrams[train_idx], train_word_ngrams[train_idx]]
)
X_val_final = hstack(
    [val_stylo_scaled, char_vectorizer.transform(train_df["cleaned_text"].values[val_idx]), word_vectorizer.transform(train_df["cleaned_text"].values[val_idx])]
)
X_test_final = hstack([test_stylo_scaled, test_char_ngrams, test_word_ngrams])

print(f"Final training features shape: {X_train_final.shape}")
print(f"Final validation features shape: {X_val_final.shape}")
print(f"Final test features shape: {X_test_final.shape}")

joblib.dump(
    {
        "X_train": X_train_final,
        "X_val": X_val_final,
        "X_test": X_test_final,
        "y_train": train_labels,
        "y_val": val_labels,
        "label_encoder": label_encoder,
        "char_vectorizer": char_vectorizer,
        "word_vectorizer": word_vectorizer,
        "scaler": scaler,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_ids": test_ids,
    },
    "./working/processed_data.pkl",
)

# Quick baseline
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    max_iter=1000, multi_class="multinomial", C=1.0, random_state=42
)
clf.fit(X_train_final, train_labels)
val_preds_bl = clf.predict_proba(X_val_final)
val_log_loss_bl = log_loss(val_labels, val_preds_bl)
print(f"Validation Log Loss (Logistic Regression baseline): {val_log_loss_bl:.5f}")

# ============================================================
# 2. MODEL DESIGN - ModernBERT with MultiPoolingHead
# ============================================================
class DropPath(nn.Module):
    """Stochastic Depth / DropPath module for regularization."""
    def __init__(self, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with uniform gamma for all classes and no class weight scaling."""
    def __init__(self, class_counts=None, gamma=3.0, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, pred, target):
        # Apply label smoothing
        num_classes = pred.size(1)
        smoothed_target = torch.full_like(pred, self.smoothing / (num_classes - 1))
        smoothed_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)

        # Focal loss: - (1 - pt)^gamma * log(pt)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy with label smoothing (no class weights)
        ce_loss = -log_probs * smoothed_target
        ce_loss = ce_loss.sum(dim=-1)

        loss = focal_weight * ce_loss
        return loss.mean()

class MultiPoolingHead(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=3, dropout_rate=0.3, droppath_rate=0.1):
        super().__init__()
        self.droppath = DropPath(drop_rate=droppath_rate)
        self.cls_proj = nn.Linear(hidden_size, hidden_size)
        self.mean_proj = nn.Linear(hidden_size, hidden_size)
        self.max_proj = nn.Linear(hidden_size, hidden_size)
        self.attention_weights = nn.Linear(hidden_size, 1)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size // 4, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [
            self.cls_proj,
            self.mean_proj,
            self.max_proj,
            self.attention_weights,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, last_hidden_state, attention_mask=None):
        # Apply DropPath to the last_hidden_state to regularize
        last_hidden_state = self.droppath(last_hidden_state)

        cls_emb = last_hidden_state[:, 0, :]
        cls_pooled = F.gelu(self.cls_proj(cls_emb))

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            mean_emb = (last_hidden_state * mask_expanded).sum(
                dim=1
            ) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            mean_emb = last_hidden_state.mean(dim=1)
        mean_pooled = F.gelu(self.mean_proj(mean_emb))

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            mask_expanded = (1.0 - mask_expanded) * -1e4
            max_emb = last_hidden_state.masked_fill(mask_expanded.bool(), -1e4).max(dim=1)[0]
        else:
            max_emb = last_hidden_state.max(dim=1)[0]
        max_pooled = F.gelu(self.max_proj(max_emb))

        attn_scores = self.attention_weights(last_hidden_state)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_emb = (attn_weights * last_hidden_state).sum(dim=1)
        attn_pooled = F.gelu(self.mean_proj(attn_emb))

        hidden_size = self.cls_proj.out_features  # or self.hidden_size
        combined = torch.cat([cls_pooled, mean_pooled, max_pooled, attn_pooled], dim=-1)
        gates = self.gate_network(combined)
        gated_cls = gates[:, :hidden_size] * cls_pooled
        gated_mean = gates[:, hidden_size:2*hidden_size] * mean_pooled
        gated_max = gates[:, 2*hidden_size:3*hidden_size] * max_pooled
        gated_attn = gates[:, 3*hidden_size:] * attn_pooled

        logits_proj_input = torch.cat([gated_cls, gated_mean, gated_max, gated_attn], dim=-1)
        logits_proj_input = self.droppath(logits_proj_input)
        logits = self.classifier(logits_proj_input)
        return logits

class ModernBERTAuthorClassifier(nn.Module):
    def __init__(
        self,
        model_id="answerdotai/ModernBERT-large",
        num_classes=3,
        dropout_rate=0.3,
        freeze_backbone=True,
        droppath_rate=0.1,
    ):
        super().__init__()
        self.backbone = ModernBertModel.from_pretrained(model_id)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier_head = MultiPoolingHead(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            droppath_rate=droppath_rate,
        )

    def unfreeze_stage(self, stage):
        """Progressive unfreezing of backbone.
        stage 0: freeze entire backbone (only head trainable)
        stage 1: unfreeze last 8 transformer layers
        stage 2: unfreeze entire backbone
        """
        # First freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        if stage >= 1:
            # Unfreeze last 8 layers
            # ModernBERT uses 'transformer_layer' list inside backbone
            if hasattr(self.backbone, 'transformer_layer'):
                layers = self.backbone.transformer_layer
                num_layers = len(layers)
                for i in range(max(0, num_layers - 8), num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            elif hasattr(self.backbone, 'layer'):
                layers = self.backbone.layer
                num_layers = len(layers)
                for i in range(max(0, num_layers - 8), num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            elif hasattr(self.backbone, 'encoder'):
                if hasattr(self.backbone.encoder, 'layer'):
                    layers = self.backbone.encoder.layer
                    num_layers = len(layers)
                    for i in range(max(0, num_layers - 8), num_layers):
                        for param in layers[i].parameters():
                            param.requires_grad = True
                elif hasattr(self.backbone.encoder, 'transformer_layer'):
                    layers = self.backbone.encoder.transformer_layer
                    num_layers = len(layers)
                    for i in range(max(0, num_layers - 8), num_layers):
                        for param in layers[i].parameters():
                            param.requires_grad = True
                else:
                    # Fallback: just unfreeze all backbone when we can't find layers
                    for param in self.backbone.parameters():
                        param.requires_grad = True
            else:
                # Fallback: just unfreeze all backbone when we can't find layers
                for param in self.backbone.parameters():
                    param.requires_grad = True
        if stage >= 2:
            # Unfreeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier_head(last_hidden_state, attention_mask)
        return logits

def get_criterion(class_counts=None):
    return AdaptiveFocalLoss(class_counts=class_counts, gamma=3.0, smoothing=0.1)

def get_optimizer(model, learning_rate=2e-5, weight_decay=0.01):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate},
        ],
        weight_decay=weight_decay,
    )
    return optimizer

def get_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )
    return scheduler

def create_model(num_classes=3, dropout_rate=0.3, freeze_backbone=True, droppath_rate=0.1):
    model_id = "answerdotai/ModernBERT-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ModernBERTAuthorClassifier(
        model_id=model_id,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    # Override the MultiPoolingHead with one that includes DropPath
    model.classifier_head = MultiPoolingHead(
        hidden_size=model.hidden_size,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        droppath_rate=droppath_rate,
    )
    model.classifier_head = model.classifier_head.to(model.backbone.device if hasattr(model.backbone, 'device') else 'cpu')
    return model, tokenizer

# ============================================================
# EMA (Exponential Moving Average) utilities
# ============================================================
class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

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
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

# ============================================================
# 3. TRAINING & EVALUATION
# ============================================================
class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Static synonym dictionary for SSMix-inspired augmentation
        self.synonym_dict = {
            "good": ["great", "fine", "excellent", "superb", "quality"],
            "bad": ["poor", "terrible", "awful", "inferior", "unpleasant"],
            "big": ["large", "huge", "enormous", "massive", "great"],
            "small": ["tiny", "little", "miniature", "compact", "petite"],
            "beautiful": ["lovely", "gorgeous", "stunning", "attractive", "pretty"],
            "important": ["crucial", "vital", "essential", "significant", "key"],
            "difficult": ["hard", "challenging", "tough", "complex", "demanding"],
            "easy": ["simple", "effortless", "smooth", "straightforward", "basic"],
            "happy": ["glad", "joyful", "cheerful", "delighted", "pleased"],
            "sad": ["unhappy", "sorrowful", "gloomy", "melancholy", "somber"],
            "strange": ["odd", "peculiar", "weird", "unusual", "bizarre"],
            "old": ["ancient", "aged", "elderly", "vintage", "antique"],
            "new": ["fresh", "modern", "recent", "current", "novel"],
            "interesting": ["engaging", "fascinating", "compelling", "intriguing"],
            "terrible": ["horrible", "dreadful", "awful", "appalling"],
            "quiet": ["silent", "still", "calm", "peaceful", "tranquil"],
            "dark": ["dim", "gloomy", "shadowy", "murky", "black"],
            "light": ["bright", "luminous", "radiant", "glowing", "vivid"],
            "cold": ["chilly", "frigid", "icy", "freezing", "frosty"],
            "hot": ["warm", "scorching", "blazing", "boiling", "fiery"],
            "fast": ["quick", "rapid", "swift", "speedy", "fleeting"],
            "slow": ["gradual", "steady", "sluggish", "leisurely"],
            "strong": ["powerful", "robust", "sturdy", "solid", "firm"],
            "weak": ["feeble", "fragile", "frail", "delicate", "slight"],
            "true": ["real", "genuine", "authentic", "actual", "valid"],
            "false": ["fake", "phony", "counterfeit", "fraudulent", "invalid"],
            "very": ["extremely", "highly", "deeply", "intensely", "profoundly"],
            "really": ["truly", "genuinely", "actually", "indeed", "certainly"],
            "always": ["constantly", "continually", "perpetually", "ever"],
            "never": ["not ever", "at no time", "not at all"],
            "many": ["numerous", "multiple", "countless", "abundant", "plenty"],
            "few": ["scarce", "sparse", "limited", "minimal", "rare"],
            "like": ["enjoy", "appreciate", "admire", "cherish"],
            "love": ["adore", "treasure", "prize", "cherish", "care for"],
            "hate": ["despise", "loathe", "detest", "abhor", "scorn"],
            "think": ["believe", "consider", "ponder", "reflect", "reckon"],
            "know": ["understand", "comprehend", "realize", "recognize", "grasp"],
            "see": ["view", "spot", "observe", "perceive", "witness"],
            "look": ["stare", "gaze", "peer", "glance", "observe"],
            "feel": ["sense", "perceive", "experience", "detect"],
            "great": ["superb", "magnificent", "wonderful", "extraordinary"],
            "little": ["slight", "minute", "negligible", "trivial", "minor"],
            "much": ["greatly", "substantially", "considerably", "vastly"],
            "well": ["properly", "adequately", "thoroughly", "satisfactorily"],
            "hard": ["difficult", "strenuous", "laborious", "arduous", "tough"],
            "long": ["lengthy", "extended", "prolonged", "interminable"],
            "short": ["brief", "concise", "succinct", "compact", "abridged"],
            "high": ["elevated", "lofty", "towering", "soaring", "steep"],
            "low": ["inferior", "substandard", "inferior", "meager", "scant"],
            "first": ["initial", "primary", "foremost", "principal", "leading"],
            "last": ["final", "ultimate", "concluding", "closing"],
            "next": ["following", "subsequent", "ensuing", "later"],
            "more": ["additional", "further", "extra", "supplementary"],
            "most": ["utmost", "maximum", "greatest", "supreme"],
            "own": ["personal", "private", "individual", "distinctive"],
            "other": ["different", "alternative", "distinct", "separate", "various"],
            "same": ["identical", "equivalent", "equal", "matching", "alike"],
            "such": ["similar", "comparable", "parallel", "analogous"],
            "just": ["exactly", "precisely", "simply", "merely", "fairly"],
            "some": ["several", "various", "assorted", "diverse", "numerous"],
            "any": ["anyone", "anybody", "anything", "whichever"],
            "each": ["every", "all", "respective", "individual"],
            "which": ["that", "what", "whichever", "whatever"],
            "who": ["that", "which", "whom", "whose"],
            "where": ["wherever", "whence", "wherein", "whither"],
            "when": ["whenever", "while", "whilst", "as"],
            "how": ["in what way", "by what means", "through what"],
            "what": ["which", "whatever", "that which", "whichever"],
            "there": ["yonder", "therein", "thither", "yonder"],
            "here": ["nearby", "hereabouts", "herein", "hereinbefore"],
            "then": ["subsequently", "afterward", "later", "thereupon"],
            "now": ["currently", "presently", "immediately", "instantly"],
            "over": ["above", "beyond", "across", "throughout"],
            "under": ["beneath", "below", "underneath", "subordinate"],
            "very": ["extremely", "exceedingly", "immensely", "remarkably"],
            "so": ["thus", "therefore", "consequently", "accordingly"],
            "too": ["also", "likewise", "excessively", "overly"],
            "but": ["however", "yet", "nevertheless", "nonetheless", "though"],
            "and": ["also", "plus", "along with", "together with"],
            "or": ["otherwise", "alternatively", "either"],
            "if": ["whether", "whenever", "provided that", "assuming"],
            "because": ["since", "as", "for", "due to", "owing to"],
            "about": ["approximately", "regarding", "concerning", "around"],
            "into": ["inside", "within", "toward", "through"],
            "through": ["via", "by", "throughout", "across"],
            "during": ["throughout", "through", "for the duration"],
            "before": ["prior to", "earlier than", "ahead of", "preceding"],
            "after": ["following", "subsequent to", "later than", "beyond"],
            "above": ["over", "higher than", "superior to", "prior to"],
            "below": ["under", "beneath", "lower than", "inferior to"],
            "between": ["among", "mid", "amid", "in the middle"],
            "through": ["via", "by means of", "by way of", "by virtue of"],
            "against": ["opposed to", "contrary to", "versus", "counter"],
            "without": ["excluding", "lacking", "devoid of", "minus"],
            "within": ["inside", "in", "enclosed by", "inside of"],
            "along": ["beside", "adjacent to", "next to", "parallel to"],
            "among": ["amid", "between", "surrounded by", "in the midst of"],
            "upon": ["on", "onto", "above", "on top of"],
            "across": ["over", "through", "spanning", "covering"],
            "behind": ["after", "following", "beyond", "at the back of"],
            "beyond": ["past", "after", "over", "outside"],
            "toward": ["towards", "in the direction of", "heading for"],
            "around": ["about", "approximately", "roughly", "nearly"],
            "down": ["below", "lower", "downward", "beneath"],
            "off": ["away", "separated", "removed", "apart from"],
            "up": ["upward", "higher", "above", "overhead"],
            "out": ["outside", "outdoors", "without", "external"],
            "in": ["inside", "within", "into", "indoors"],
            "on": ["upon", "atop", "resting on", "situated on"],
            "at": ["by", "near", "beside", "adjacent to"],
            "by": ["via", "through", "by means of", "using"],
            "with": ["accompanied by", "together with", "alongside", "including"],
            "for": ["on behalf of", "representing", "in favor of", "during"],
            "to": ["toward", "in the direction of", "as far as", "until"],
            "from": ["out of", "originating from", "starting from", "away from"],
            "as": ["like", "similar to", "in the role of", "serving as"],
            "it": ["this", "that", "the thing", "the item"],
            "its": ["its own", "belonging to it", "of it"],
            "this": ["this one", "this thing", "the aforementioned", "the present"],
            "that": ["that one", "that thing", "the aforementioned", "the former"],
            "these": ["these ones", "these things", "the aforementioned"],
            "those": ["those ones", "those things", "the aforementioned"],
            "i": ["i myself", "the author", "yours truly", "me"],
            "you": ["yourselves", "you all", "you guys", "you people"],
            "he": ["himself", "that man", "that person", "the gentleman"],
            "she": ["herself", "that woman", "that person", "the lady"],
            "we": ["ourselves", "us", "we all", "all of us"],
            "they": ["them", "themselves", "those people", "others"],
            "me": ["myself", "yours truly", "the author"],
            "him": ["himself", "that man", "the gentleman"],
            "her": ["herself", "that woman", "the lady"],
            "us": ["ourselves", "we", "all of us"],
            "them": ["themselves", "those people", "the others"],
            "my": ["my own", "belonging to me", "mine"],
            "your": ["your own", "belonging to you", "yours"],
            "his": ["his own", "belonging to him", "his very own"],
            "her": ["her own", "belonging to her", "hers"],
            "our": ["our own", "belonging to us", "ours"],
            "their": ["their own", "belonging to them", "theirs"],
            "itself": ["its very self", "the thing itself"],
            "myself": ["i myself", "me personally", "yours truly"],
            "yourself": ["you yourself", "you personally", "you in person"],
            "himself": ["he himself", "that person himself"],
            "herself": ["she herself", "that lady herself"],
            "ourselves": ["we ourselves", "us personally"],
            "themselves": ["they themselves", "the very people"],
            "man": ["gentleman", "person", "individual", "fellow", "male"],
            "woman": ["lady", "person", "individual", "female", "gentlewoman"],
            "person": ["individual", "human", "being", "someone"],
            "people": ["persons", "individuals", "humans", "folks"],
            "child": ["kid", "youngster", "boy", "girl", "infant", "youth"],
            "children": ["kids", "youngsters", "boys", "girls", "youth"],
            "thing": ["object", "item", "article", "entity", "matter"],
            "way": ["method", "manner", "approach", "means", "technique"],
            "place": ["location", "spot", "site", "area", "position"],
            "time": ["period", "duration", "moment", "occasion", "era"],
            "year": ["twelvemonth", "calendar year", "annual period"],
            "day": ["daytime", "twenty-four hours", "period"],
            "night": ["nighttime", "evening", "darkness", "after dark"],
            "world": ["earth", "globe", "planet", "universe", "realm"],
            "life": ["existence", "living", "being", "lifetime"],
            "hand": ["paw", "fist", "palm", "grasp", "grip"],
            "eye": ["eyeball", "ocular", "vision", "sight", "gaze"],
            "face": ["countenance", "visage", "features", "expression"],
            "head": ["cranium", "skull", "mind", "brain", "intellect"],
            "heart": ["core", "center", "essence", "soul", "spirit"],
            "mind": ["intellect", "brain", "understanding", "consciousness"],
            "body": ["figure", "form", "physique", "frame", "build"],
            "house": ["home", "dwelling", "residence", "abode", "lodging"],
            "room": ["chamber", "space", "area", "compartment", "hall"],
            "door": ["entrance", "gateway", "portal", "opening", "threshold"],
            "window": ["pane", "opening", "casement", "transom"],
            "table": ["desk", "worktable", "counter", "board"],
            "chair": ["seat", "bench", "stool", "armchair", "recliner"],
            "book": ["volume", "tome", "work", "publication", "text"],
            "word": ["term", "expression", "vocable", "utterance"],
            "letter": ["epistle", "note", "correspondence", "missive"],
            "story": ["tale", "narrative", "account", "chronicle", "yarn"],
            "name": ["title", "designation", "label", "appellation", "moniker"],
            "part": ["portion", "piece", "segment", "section", "component"],
            "kind": ["type", "sort", "variety", "class", "category"],
            "hand": ["help", "assistance", "aid", "support", "service"],
            "side": ["edge", "border", "flank", "margin", "boundary"],
            "line": ["row", "column", "queue", "file", "string"],
            "end": ["termination", "conclusion", "finish", "closure"],
            "beginning": ["start", "commencement", "outset", "threshold"],
            "middle": ["center", "core", "midpoint", "interior", "heart"],
            "top": ["peak", "summit", "apex", "crown", "head"],
            "bottom": ["base", "foundation", "foot", "underneath"],
            "front": ["forehead", "vanguard", "leading edge", "foremost"],
            "back": ["rear", "hind", "posterior", "reverse side"],
            "right": ["correct", "proper", "accurate", "appropriate", "fitting"],
            "wrong": ["incorrect", "mistaken", "erroneous", "false", "inaccurate"],
            "true": ["factual", "accurate", "genuine", "real", "authentic"],
            "false": ["untrue", "incorrect", "unreal", "deceptive", "misleading"],
            "full": ["complete", "entire", "total", "whole", "filled"],
            "empty": ["vacant", "void", "bare", "hollow", "unfilled"],
            "young": ["youthful", "juvenile", "junior", "immature", "new"],
            "dear": ["beloved", "cherished", "precious", "valued", "darling"],
            "poor": ["impoverished", "needy", "destitute", "indigent"],
            "rich": ["wealthy", "affluent", "prosperous", "well-off"],
            "free": ["liberated", "independent", "unrestricted", "unbound"],
            "clear": ["obvious", "apparent", "evident", "distinct", "plain"],
            "dark": ["unlit", "dim", "murky", "gloomy", "shadowy"],
            "bright": ["shining", "luminous", "radiant", "brilliant", "vivid"],
            "deep": ["profound", "intense", "bottomless", "abysmal"],
            "wide": ["broad", "extended", "extensive", "spacious"],
            "thin": ["slender", "narrow", "fine", "delicate", "slim"],
            "thick": ["dense", "heavy", "substantial", "bulky", "solid"],
            "near": ["close", "adjacent", "neighboring", "proximate", "nigh"],
            "far": ["distant", "remote", "faraway", "outlying", "afar"],
            "dear": ["darling", "beloved", "sweetheart", "honey"],
            "goodbye": ["farewell", "bye", "adieu", "toodle-oo"],
            "hello": ["hi", "greetings", "salutations", "howdy"],
            "yes": ["yeah", "yep", "indeed", "certainly", "surely"],
            "no": ["nay", "not", "negative", "never"],
            "please": ["kindly", "pray", "if you please", "prithee"],
            "sorry": ["apologetic", "regretful", "remorseful", "contrite"],
            "thank": ["gratitude", "thanks", "appreciation", "acknowledgment"],
            "well": ["good", "satisfactorily", "fine", "adequately"],
            "still": ["yet", "nevertheless", "quiet", "motionless"],
            "even": ["level", "smooth", "flat", "uniform"],
            "just": ["fair", "equitable", "honest", "righteous"],
            "only": ["alone", "singular", "sole", "exclusive"],
            "very": ["same", "selfsame", "identical", "exact", "precise"],
            "too": ["overly", "excessively", "unduly", "immoderately"],
            "much": ["greatly", "considerably", "substantially", "vastly"],
            "quite": ["rather", "fairly", "reasonably", "relatively"],
            "pretty": ["fairly", "quite", "rather", "somewhat", "moderately"],
            "rather": ["quite", "somewhat", "fairly", "relatively"],
            "somewhat": ["rather", "fairly", "quite", "moderately"],
            "almost": ["nearly", "practically", "virtually", "approximately"],
            "hardly": ["barely", "scarcely", "rarely", "seldom"],
            "nearly": ["almost", "approximately", "virtually", "practically"],
            "simply": ["plainly", "clearly", "just", "merely", "only"],
            "really": ["truly", "genuinely", "actually", "indeed", "certainly"],
            "finally": ["ultimately", "eventually", "at last", "in conclusion"],
            "suddenly": ["abruptly", "unexpectedly", "all at once", "without warning"],
            "immediately": ["instantly", "right away", "at once", "directly"],
            "quickly": ["swiftly", "rapidly", "speedily", "hastily", "fast"],
            "slowly": ["gradually", "leisurely", "unhurriedly", "tardily"],
            "carefully": ["cautiously", "warily", "gingerly", "studiously"],
            "eagerly": ["enthusiastically", "readily", "willingly", "keenly"],
            "happily": ["joyfully", "cheerfully", "blithely", "merrily"],
            "sadly": ["unfortunately", "regrettably", "mournfully", "dolefully"],
            "certainly": ["definitely", "unquestionably", "indubitably", "assuredly"],
            "probably": ["likely", "presumably", "doubtless", "in all probability"],
            "perhaps": ["maybe", "possibly", "perchance", "it may be"],
            "indeed": ["truly", "certainly", "undoubtedly", "verily"],
            "therefore": ["hence", "thus", "consequently", "accordingly"],
            "however": ["nevertheless", "nonetheless", "yet", "still", "though"],
            "meanwhile": ["meantime", "simultaneously", "concurrently"],
            "moreover": ["furthermore", "additionally", "besides", "also"],
            "nevertheless": ["nonetheless", "however", "yet", "still", "though"],
            "consequently": ["therefore", "hence", "thus", "accordingly"],
            "fortunately": ["luckily", "happily", "providentially"],
            "unfortunately": ["regrettably", "sadly", "unhappily", "alas"],
            "perhaps": ["maybe", "perchance", "possibly", "mayhap"],
            "indeed": ["verily", "truly", "certainly", "of course"],
            "never": ["not ever", "at no time", "nevermore", "nary"],
            "always": ["forever", "evermore", "perpetually", "unceasingly"],
            "often": ["frequently", "regularly", "repeatedly", "commonly"],
            "seldom": ["rarely", "infrequently", "scarcely", "hardly ever"],
            "sometimes": ["occasionally", "at times", "now and then", "from time to time"],
            "usually": ["generally", "normally", "typically", "ordinarily", "customarily"],
            "again": ["once more", "anew", "afresh", "again and again"],
            "also": ["too", "likewise", "as well", "additionally", "further"],
            "almost": ["nearly", "virtually", "practically", "well-nigh"],
            "already": ["by now", "previously", "beforehand", "already"],
            "always": ["ever", "forever", "perpetually", "unceasingly"],
            "anyway": ["anyhow", "at any rate", "in any case", "regardless"],
            "besides": ["moreover", "furthermore", "in addition", "also"],
            "certainly": ["definitely", "surely", "unquestionably", "assuredly"],
            "clearly": ["obviously", "evidently", "apparently", "manifestly"],
            "closely": ["intently", "carefully", "attentively", "watchfully"],
            "constantly": ["continually", "persistently", "incessantly", "unceasingly"],
            "deeply": ["profoundly", "intensely", "keenly", "acutely"],
            "definitely": ["certainly", "surely", "unquestionably", "absolutely"],
            "directly": ["immediately", "instantly", "straightaway", "right away"],
            "especially": ["particularly", "notably", "specifically", "mainly"],
            "even": ["still", "yet", "all the more", "even more"],
            "ever": ["always", "forever", "at any time", "evermore"],
            "exactly": ["precisely", "accurately", "correctly", "strictly"],
            "finally": ["ultimately", "eventually", "in the end", "at last"],
            "firmly": ["solidly", "securely", "tightly", "steadfastly"],
            "first": ["initially", "firstly", "to begin with", "originally"],
            "fully": ["completely", "entirely", "wholly", "thoroughly"],
            "generally": ["usually", "typically", "normally", "commonly"],
            "gently": ["softly", "tenderly", "lightly", "mildly"],
            "gladly": ["happily", "joyfully", "cheerfully", "readily"],
            "gradually": ["slowly", "progressively", "steadily", "bit by bit"],
            "greatly": ["highly", "very much", "immensely", "enormously"],
            "hardly": ["barely", "scarcely", "rarely", "faintly"],
            "heavily": ["intensely", "severely", "greatly", "deeply"],
            "highly": ["extremely", "very", "greatly", "immensely", "deeply"],
            "immediately": ["instantly", "at once", "right away", "directly"],
            "increasingly": ["progressively", "ever more", "growing", "more and more"],
            "instantly": ["immediately", "at once", "right away", "directly"],
            "largely": ["mostly", "mainly", "primarily", "chiefly"],
            "lightly": ["gently", "softly", "delicately", "mildly"],
            "likely": ["probably", "presumably", "doubtless", "in all likelihood"],
            "literally": ["exactly", "precisely", "actually", "truly"],
            "merely": ["only", "just", "simply", "barely"],
            "mighty": ["powerfully", "strongly", "forcefully", "vigorously"],
            "mildly": ["slightly", "somewhat", "rather", "fairly"],
            "mostly": ["mainly", "largely", "chiefly", "primarily"],
            "namely": ["specifically", "that is", "i.e.", "in other words"],
            "naturally": ["certainly", "of course", "obviously", "understandably"],
            "nearly": ["almost", "approximately", "practically", "virtually"],
            "neatly": ["tidily", "orderly", "cleanly", "precisely"],
            "necessarily": ["inevitably", "unavoidably", "of necessity", "surely"],
            "nicely": ["pleasantly", "agreeably", "delightfully", "charmingly"],
            "normally": ["usually", "generally", "typically", "ordinarily"],
            "obviously": ["clearly", "evidently", "apparently", "plainly"],
            "occasionally": ["sometimes", "now and then", "from time to time", "periodically"],
            "openly": ["frankly", "candidly", "honestly", "directly"],
            "originally": ["initially", "first", "at first", "in the beginning"],
            "particularly": ["especially", "notably", "specifically", "in particular"],
            "perfectly": ["flawlessly", "impeccably", "exquisitely", "superbly"],
            "personally": ["individually", "privately", "in person", "intimately"],
            "plainly": ["clearly", "obviously", "evidently", "transparently"],
            "pleasantly": ["agreeably", "delightfully", "enjoyably", "nicely"],
            "poorly": ["badly", "unsatisfactorily", "inadequately", "insufficiently"],
            "possibly": ["maybe", "perhaps", "perchance", "it is possible"],
            "precisely": ["exactly", "accurately", "correctly", "strictly"],
            "presently": ["currently", "at present", "now", "immediately"],
            "presumably": ["likely", "probably", "doubtless", "assumably"],
            "previously": ["before", "earlier", "formerly", "prior"],
            "primarily": ["mainly", "mostly", "chiefly", "principally"],
            "principally": ["mainly", "primarily", "mostly", "chiefly"],
            "privately": ["personally", "confidentially", "in private", "secretly"],
            "probably": ["likely", "presumably", "doubtless", "in all probability"],
            "promptly": ["immediately", "quickly", "swiftly", "at once"],
            "properly": ["correctly", "appropriately", "suitably", "accurately"],
            "quickly": ["rapidly", "swiftly", "fast", "speedily"],
            "quietly": ["silently", "softly", "noiselessly", "calmly"],
            "rapidly": ["quickly", "swiftly", "fast", "speedily"],
            "rarely": ["seldom", "infrequently", "scarcely", "hardly ever"],
            "readily": ["easily", "willingly", "quickly", "promptly"],
            "really": ["truly", "actually", "genuinely", "indeed"],
            "recently": ["newly", "lately", "freshly", "of late"],
            "recklessly": ["rashly", "foolhardily", "carelessly", "heedlessly"],
            "regularly": ["routinely", "frequently", "habitually", "periodically"],
            "relatively": ["comparatively", "somewhat", "rather", "fairly"],
            "reluctantly": ["unwillingly", "hesitantly", "grudgingly", "resentfully"],
            "remarkably": ["notably", "strikingly", "impressively", "extraordinarily"],
            "repeatedly": ["again and again", "over and over", "frequently", "time after time"],
            "rightly": ["correctly", "justly", "fairly", "properly"],
            "roughly": ["approximately", "about", "around", "nearly"],
            "routinely": ["regularly", "habitually", "customarily", "ordinarily"],
            "sadly": ["unfortunately", "regrettably", "mournfully", "dolefully"],
            "safely": ["securely", "soundly", "without mishap", "without incident"],
            "scarcely": ["barely", "hardly", "rarely", "seldom"],
            "secretly": ["privately", "covertly", "stealthily", "surreptitiously"],
            "seldom": ["rarely", "infrequently", "scarcely", "hardly ever"],
            "separately": ["individually", "apart", "singly", "distinctly"],
            "seriously": ["gravely", "solemnly", "earnestly", "soberly"],
            "severely": ["badly", "seriously", "critically", "grievously"],
            "sharply": ["abruptly", "keenly", "acutely", "intensely"],
            "shortly": ["briefly", "soon", "presently", "in a short time"],
            "significantly": ["notably", "substantially", "considerably", "remarkably"],
            "silently": ["quietly", "noiselessly", "soundlessly", "speechlessly"],
            "similarly": ["likewise", "correspondingly", "in like manner", "also"],
            "simply": ["plainly", "clearly", "straightforwardly", "easily"],
            "sincerely": ["honestly", "truly", "genuinely", "earnestly"],
            "slightly": ["somewhat", "a little", "marginally", "faintly"],
            "slowly": ["gradually", "leisurely", "unhurriedly", "steadily"],
            "smoothly": ["evenly", "fluently", "uninterruptedly", "effortlessly"],
            "softly": ["gently", "tenderly", "lightly", "mildly"],
            "solemnly": ["gravely", "seriously", "soberly", "earnestly"],
            "solidly": ["firmly", "strongly", "securely", "substantially"],
            "somehow": ["some way", "in some way", "by some means", "somewise"],
            "sometimes": ["occasionally", "at times", "now and then", "from time to time"],
            "somewhat": ["rather", "fairly", "quite", "moderately"],
            "soon": ["shortly", "presently", "before long", "in the near future"],
            "sorely": ["greatly", "deeply", "severely", "intensely"],
            "specifically": ["particularly", "notably", "explicitly", "precisely"],
            "speedily": ["quickly", "rapidly", "swiftly", "fast"],
            "steadily": ["gradually", "progressively", "continuously", "unwaveringly"],
            "stealthily": ["secretly", "covertly", "surreptitiously", "furtively"],
            "sternly": ["strictly", "harshly", "firmly", "severely"],
            "strangely": ["peculiarly", "oddly", "unusually", "curiously"],
            "strictly": ["precisely", "exactly", "rigorously", "severely"],
            "strongly": ["firmly", "powerfully", "forcefully", "vigorously"],
            "substantially": ["considerably", "significantly", "greatly", "largely"],
            "successfully": ["effectively", "efficiently", "competently", "capably"],
            "suddenly": ["abruptly", "unexpectedly", "all of a sudden", "without warning"],
            "sufficiently": ["adequately", "enough", "satisfactorily", "properly"],
            "suitably": ["appropriately", "properly", "fittingly", "correctly"],
            "surely": ["certainly", "definitely", "undoubtedly", "assuredly"],
            "surprisingly": ["unexpectedly", "astonishingly", "astoundingly", "remarkably"],
            "swiftly": ["quickly", "rapidly", "fast", "speedily"],
            "tenderly": ["lovingly", "affectionately", "gently", "softly"],
            "thoroughly": ["completely", "fully", "entirely", "exhaustively"],
            "thus": ["therefore", "hence", "consequently", "accordingly"],
            "tightly": ["firmly", "securely", "snugly", "compactly"],
            "totally": ["completely", "entirely", "fully", "absolutely"],
            "truly": ["really", "genuinely", "sincerely", "honestly"],
            "typically": ["generally", "usually", "normally", "ordinarily"],
            "ultimately": ["finally", "eventually", "in the end", "at last"],
            "unanimously": ["jointly", "by common consent", "without dissent"],
            "undoubtedly": ["certainly", "unquestionably", "indubitably", "without doubt"],
            "unexpectedly": ["suddenly", "abruptly", "surprisingly", "without warning"],
            "universally": ["everywhere", "worldwide", "globally", "by all"],
            "unusually": ["remarkably", "notably", "extraordinarily", "exceptionally"],
            "urgently": ["immediately", "pressingly", "imperatively", "critically"],
            "usefully": ["helpfully", "practically", "effectively", "beneficially"],
            "usually": ["normally", "generally", "typically", "ordinarily"],
            "utterly": ["totally", "completely", "absolutely", "entirely"],
            "vaguely": ["indistinctly", "obscurely", "ambiguously", "faintly"],
            "vastly": ["enormously", "immensely", "tremendously", "greatly"],
            "verbally": ["orally", "in words", "by word of mouth", "spokenly"],
            "very": ["extremely", "highly", "deeply", "immensely", "remarkably"],
            "vigorously": ["strongly", "forcefully", "energetically", "powerfully"],
            "violently": ["forcefully", "wildly", "furiously", "passionately"],
            "virtually": ["almost", "nearly", "practically", "essentially"],
            "vividly": ["clearly", "strikingly", "graphically", "intensely"],
            "voluntarily": ["willingly", "freely", "by choice", "of one's own accord"],
            "warmly": ["cordially", "heartily", "affectionately", "genially"],
            "weakly": ["feebly", "faintly", "fleetingly", "slenderly"],
            "wearily": ["tiredly", "exhaustedly", "fatiguedly", "doggedly"],
            "wholly": ["completely", "entirely", "totally", "fully"],
            "wickedly": ["evilly", "viciously", "immorally", "sinfully"],
            "widely": ["broadly", "extensively", "universally", "commonly"],
            "willingly": ["voluntarily", "readily", "gladly", "eagerly"],
            "wisely": ["prudently", "sensibly", "astutely", "shrewdly"],
            "wonderfully": ["marvelously", "remarkably", "superbly", "splendidly"],
            "yearly": ["annually", "per annum", "each year", "year after year"],
            "youthfully": ["boyishly", "girlishly", "freshly", "vigorously"]
        }

    def ss_mix_augment(self, text, p_augment=0.5, aug_frac=0.15):
        """Apply SSMix-inspired augmentation: synonym replacement and random word insertion on aug_frac of tokens."""
        if np.random.random() > p_augment:
            return text

        words = text.split()
        if len(words) < 5:
            return text

        num_tokens = max(1, int(len(words) * aug_frac))

        # Get candidates for synonym replacement
        word_list = list(enumerate(words))
        candidates = [(i, w.lower()) for i, w in word_list if w.lower() in self.synonym_dict]

        if not candidates:
            return text

        num_replace = min(num_tokens, len(candidates))
        replace_indices = np.random.choice(len(candidates), num_replace, replace=False)
        selected = [candidates[idx] for idx in replace_indices]

        for orig_idx, orig_word_lower in selected:
            synonyms = self.synonym_dict[orig_word_lower]
            chosen_synonym = np.random.choice(synonyms)
            # Preserve capitalization
            if words[orig_idx][0].isupper():
                chosen_synonym = chosen_synonym.capitalize()
            words[orig_idx] = chosen_synonym

        # Random word insertion (insert synonyms at random positions)
        unused_candidates = [c for c in candidates if c not in selected]
        if unused_candidates:
            num_insert = min(max(0, num_tokens - num_replace), len(unused_candidates))
            if num_insert > 0:
                insert_pool = [self.synonym_dict[c[1]] for c in unused_candidates[:num_insert]]
                insert_words = [np.random.choice(syns) for syns in insert_pool]
                insert_positions = sorted(np.random.choice(len(words) + 1, num_insert, replace=False))
                for pos, word in zip(insert_positions, insert_words):
                    words.insert(pos, word)

        return " ".join(words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Apply text augmentation with 50% probability
        if self.labels is not None:  # Only augment for training samples
            text = self.ss_mix_augment(text, p_augment=0.5, aug_frac=0.15)
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class CollateWithMask:
    """Collate function that has access to the tokenizer's mask token id."""
    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id

    def __call__(self, batch):
        MASK_TOKEN_ID = self.mask_token_id
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for item in batch:
            input_ids = item["input_ids"].clone()
            attention_mask = item["attention_mask"].clone()
            if item.get("labels") is not None:
                labels_list.append(item["labels"])
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        batch_output = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
        }
        if labels_list:
            batch_output["labels"] = torch.stack(labels_list)
        return batch_output

# Instantiate tokenizer first for augmentation collate function
_model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(_model_id)
augmentation_collate_fn = CollateWithMask(tokenizer.mask_token_id)

print("\n" + "=" * 60)
print("TRAINING PHASE")
print("=" * 60)

# Override FREEZE_BACKBONE for first stage (we will manage unfreezing manually)
model, tokenizer = create_model(
    num_classes=3, dropout_rate=DROPOUT_RATE, freeze_backbone=True  # Start fully frozen
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} (initially only classification head)")

train_dataset = AuthorDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = AuthorDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=augmentation_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

criterion = get_criterion()

# Compute class counts for AdaptiveFocalLoss
class_counts = []
for c in range(3):
    class_counts.append((train_labels == c).sum())
print(f"Class counts for focal loss: {class_counts}")

criterion = get_criterion(class_counts=class_counts)

num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS

# Progressive unfreezing stages:
# Stage 0 (epochs 1-2): freeze backbone, head lr=3e-5
# Stage 1 (epochs 3-5): unfreeze last 8 layers, backbone lr=1e-5
# Stage 2 (epochs 6-10): unfreeze entire backbone, backbone lr=5e-6
STAGE_EPOCHS = [2, 5, NUM_EPOCHS]  # stage transitions at epoch 3, 6

# Initialize optimizer and scheduler will be re-created each stage
class FGM:
    """Fast Gradient Method for adversarial training on embedding layer."""
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.emb_backup = {}

    def attack(self):
        # Back up original embeddings - ModernBERT stores embeddings in backbone.embeddings
        self.emb_backup = {}
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad and 'embed' in name.lower() and ('weight' in name and param.dim() == 2):
                self.emb_backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm > 0 and not torch.isnan(norm) and not torch.isinf(norm):
                        r_adv = self.epsilon * param.grad / norm
                        param.data.add_(r_adv)

    def restore(self):
        for name, param in self.model.backbone.named_parameters():
            if name in self.emb_backup:
                param.data = self.emb_backup[name]
        self.emb_backup = {}

scaler = GradScaler()

best_val_score = float("inf")
patience_counter = 0
best_model_state = None
current_stage = 0

# Initialize EMA with correct decay and ensure proper application
ema = EMA(model, decay=0.995)

# Initialize FGM with epsilon=0.5
fgm = FGM(model, epsilon=0.5)

# Initialize optimizer for stage 0 before the loop
head_lr = 3e-5
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

optimizer = AdamW(
    [
        {"params": backbone_params, "lr": 0.0},
        {"params": head_params, "lr": head_lr},
    ],
    weight_decay=WEIGHT_DECAY,
)
num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
scheduler = get_scheduler(optimizer, num_training_steps, warmup_ratio=WARMUP_RATIO)

for epoch in range(NUM_EPOCHS):
    # Determine stage before training this epoch
    new_stage = 0
    if epoch < 2:  # epochs 0,1 (1,2)
        new_stage = 0
    elif epoch < 5:  # epochs 2,3,4 (3,4,5)
        new_stage = 1
    else:  # epochs >=5 (6,7,8,...)
        new_stage = 2

    if new_stage != current_stage:
        current_stage = new_stage
        print(f"--- Transitioning to Progressive Unfreezing Stage {current_stage} ---")
        model.unfreeze_stage(current_stage)

        # Re-create optimizer with stage-specific learning rates
        if current_stage == 0:
            head_lr = 3e-5
            backbone_lr = 0.0  # frozen, no backbone params trainable
        elif current_stage == 1:
            head_lr = 3e-5
            backbone_lr = 1e-5
        else:  # stage 2
            head_lr = 3e-5
            backbone_lr = 5e-6

        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr if backbone_lr > 0 else 0.0},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        # Reset scheduler for this stage
        scheduler = get_scheduler(optimizer, num_training_steps, warmup_ratio=WARMUP_RATIO)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters after stage change: {trainable_params:,}")

        # Re-register EMA for newly unfrozen parameters (replaces shadow dict with current model params)
        ema.register()

    model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Standard forward pass
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        loss = loss / GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()

        # FGM adversarial training (only when backbone is unfrozen - stages 1 and 2)
        if current_stage >= 1:
            # First backward to get gradients on embeddings
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            # Update EMA after optimizer step
            ema.update()

            # FGM: perturb embeddings using computed gradients
            fgm.attack()
            # Re-run forward with perturbed embeddings
            with autocast():
                adv_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                adv_loss = criterion(adv_logits, labels)
            adv_loss = adv_loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(adv_loss).backward()
            # Restore original embeddings, then do the step again
            fgm.restore()

            # Override: don't do a second step here; let the normal accumulator
            # skip the step flag since we already stepped
            continue  # Skip the normal step for this iteration

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            # Update EMA after optimizer step
            ema.update()

        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    avg_train_loss = total_train_loss / len(train_loader)

    # Evaluate with both regular model and EMA model
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(logits, dim=-1)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )
    val_score = log_loss(val_targets, val_preds_normalized)

    # Also evaluate EMA model on validation set
    ema.apply_shadow()
    ema_val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            ema_val_preds.append(probs.cpu().numpy())
    ema_val_preds = np.concatenate(ema_val_preds, axis=0)
    ema_val_preds_clipped = np.clip(ema_val_preds, 1e-15, 1 - 1e-15)
    ema_val_preds_normalized = ema_val_preds_clipped / ema_val_preds_clipped.sum(axis=1, keepdims=True)
    ema_val_score = log_loss(val_targets, ema_val_preds_normalized)
    ema.restore()

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val LogLoss: {val_score:.5f} (EMA: {ema_val_score:.5f})"
    )

    # Use the best of regular vs EMA for saving
    if ema_val_score < val_score and ema_val_score < best_val_score:
        best_val_score = ema_val_score
        patience_counter = 0
        # Save regular model state BEFORE applying shadow (to avoid double-application on reload)
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_model_state, "./working/best_model.pt")
        # Apply shadow to get EMA weights and save as ema model
        ema.apply_shadow()
        best_ema_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_ema_state, "./working/best_ema_model.pt")
        ema.restore()
        print(f"  -> New best EMA model saved (LogLoss: {ema_val_score:.5f})")
    elif val_score < best_val_score:
        best_val_score = val_score
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_model_state, "./working/best_model.pt")
        print(f"  -> New best model saved (LogLoss: {val_score:.5f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Load best model (prefer EMA if available)
import os
if os.path.exists("./working/best_ema_model.pt"):
    print("Loading best EMA model for inference")
    state_dict = torch.load("./working/best_ema_model.pt", map_location="cpu")
else:
    print("Loading best regular model for inference")
    state_dict = torch.load("./working/best_model.pt", map_location="cpu")
model_state = model.state_dict()
# Filter out backbone keys if shape mismatch but still require head keys to match
filtered = {}
missing_head_keys = []
for k, v in state_dict.items():
    if k in model_state:
        if v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            print(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state[k].shape}, skipping")
    else:
        print(f"Extra key in checkpoint: {k}, skipping")
for k in model_state:
    if k not in filtered and 'classifier_head' in k:
        missing_head_keys.append(k)
if missing_head_keys:
    print(f"Warning: Missing classifier head keys: {missing_head_keys}")
model.load_state_dict(filtered, strict=False)
model = model.to(device)

# ============================================================
# 4. INFERENCE AND SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("INFERENCE PHASE")
print("=" * 60)

model.eval()

# Validation predictions
val_dataset = AuthorDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)
        val_preds.append(probs.cpu().numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_labels, val_preds_normalized)
print(f"Final Validation Score: {final_val_score}")

# Test predictions
test_dataset = AuthorDataset(
    test_texts, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds[:, 0],
        "HPL": test_preds[:, 1],
        "MWS": test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

print(f"\nFinal Validation Score: {final_val_score}")
