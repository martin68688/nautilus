import os
os.sched_setaffinity(0, {97, 98, 99, 100, 103})
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
import pickle
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
import gc
import warnings

warnings.filterwarnings("ignore")

# Download NLTK resources (quietly)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# Try to download POS tagger, but handle failure gracefully
try:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.data.find("taggers/averaged_perceptron_tagger_eng/")
    POS_TAGGER_AVAILABLE = True
except (LookupError, Exception):
    POS_TAGGER_AVAILABLE = False

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Authors: {train_df['author'].value_counts().to_dict()}")


# ============================================================
# 2. BASIC CLEANING
# ============================================================
def clean_text(text):
    """Basic text cleaning preserving important stylistic markers"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)


# ============================================================
# 3. STYLOMETRIC FEATURE ENGINEERING
# ============================================================
def extract_stylometric_features(text_series):
    """Engineer features capturing author-specific writing styles"""
    features = pd.DataFrame(index=text_series.index)

    for idx, text in enumerate(text_series):
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)

        features.loc[idx, "char_count"] = len(text)
        features.loc[idx, "word_count"] = len(words)
        features.loc[idx, "sentence_count"] = len(sentences)
        features.loc[idx, "avg_word_length"] = (
            np.mean([len(w) for w in words]) if words else 0
        )
        features.loc[idx, "avg_sentence_length"] = (
            np.mean([len(s.split()) for s in sentences]) if sentences else 0
        )
        features.loc[idx, "word_length_std"] = (
            np.std([len(w) for w in words]) if len(words) > 1 else 0
        )
        features.loc[idx, "sentence_length_std"] = (
            np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
        )

        punct_counts = {p: text.count(p) for p in ".,;:!?'\"-—()[]{}…"}
        features.loc[idx, "comma_ratio"] = punct_counts[","] / max(len(sentences), 1)
        features.loc[idx, "exclamation_ratio"] = punct_counts["!"] / max(
            len(sentences), 1
        )
        features.loc[idx, "question_ratio"] = punct_counts["?"] / max(len(sentences), 1)
        features.loc[idx, "semicolon_ratio"] = punct_counts[";"] / max(
            len(sentences), 1
        )
        features.loc[idx, "colon_ratio"] = punct_counts[":"] / max(len(sentences), 1)
        features.loc[idx, "dash_ratio"] = punct_counts["—"] + punct_counts["-"]
        features.loc[idx, "quote_ratio"] = (
            punct_counts['"'] + punct_counts["'"]
        ) / max(len(text), 1)
        features.loc[idx, "ellipsis_count"] = text.count("...")

        features.loc[idx, "capital_ratio"] = sum(1 for c in text if c.isupper()) / max(
            len(text), 1
        )
        features.loc[idx, "upper_word_ratio"] = sum(
            1 for w in words if w.isupper()
        ) / max(len(words), 1)
        features.loc[idx, "title_case_ratio"] = sum(
            1 for w in words if w.istitle()
        ) / max(len(words), 1)

        if words:
            unique_words = set(words)
            features.loc[idx, "type_token_ratio"] = len(unique_words) / len(words)
        else:
            features.loc[idx, "type_token_ratio"] = 0

        stopwords = set(nltk.corpus.stopwords.words("english"))
        stopword_count = sum(1 for w in words if w in stopwords)
        features.loc[idx, "stopword_ratio"] = stopword_count / max(len(words), 1)

        # POS tagging features (with graceful fallback)
        if words and POS_TAGGER_AVAILABLE:
            try:
                pos_tags = pos_tag(words)
                pos_counts = Counter(tag for _, tag in pos_tags)
                total_pos = len(pos_tags)
                features.loc[idx, "noun_ratio"] = (
                    pos_counts.get("NN", 0)
                    + pos_counts.get("NNS", 0)
                    + pos_counts.get("NNP", 0)
                    + pos_counts.get("NNPS", 0)
                )
                features.loc[idx, "verb_ratio"] = (
                    pos_counts.get("VB", 0)
                    + pos_counts.get("VBD", 0)
                    + pos_counts.get("VBG", 0)
                    + pos_counts.get("VBN", 0)
                    + pos_counts.get("VBP", 0)
                    + pos_counts.get("VBZ", 0)
                )
                features.loc[idx, "adj_ratio"] = (
                    pos_counts.get("JJ", 0)
                    + pos_counts.get("JJR", 0)
                    + pos_counts.get("JJS", 0)
                )
                features.loc[idx, "adv_ratio"] = (
                    pos_counts.get("RB", 0)
                    + pos_counts.get("RBR", 0)
                    + pos_counts.get("RBS", 0)
                )
                features.loc[idx, "pronoun_ratio"] = pos_counts.get(
                    "PRP", 0
                ) + pos_counts.get("PRP$", 0)
                features.loc[idx, "prep_ratio"] = pos_counts.get("IN", 0)
                features.loc[idx, "det_ratio"] = pos_counts.get("DT", 0)

                for col in [
                    "noun_ratio",
                    "verb_ratio",
                    "adj_ratio",
                    "adv_ratio",
                    "pronoun_ratio",
                    "prep_ratio",
                    "det_ratio",
                ]:
                    features.loc[idx, col] = (
                        features.loc[idx, col] / total_pos if total_pos > 0 else 0
                    )
            except Exception:
                for col in [
                    "noun_ratio",
                    "verb_ratio",
                    "adj_ratio",
                    "adv_ratio",
                    "pronoun_ratio",
                    "prep_ratio",
                    "det_ratio",
                ]:
                    features.loc[idx, col] = 0
        else:
            for col in [
                "noun_ratio",
                "verb_ratio",
                "adj_ratio",
                "adv_ratio",
                "pronoun_ratio",
                "prep_ratio",
                "det_ratio",
            ]:
                features.loc[idx, col] = 0

    return features


print("Extracting stylometric features...")
train_stylo = extract_stylometric_features(train_df["clean_text"])
test_stylo = extract_stylometric_features(test_df["clean_text"])
print(f"Stylometric features shape: {train_stylo.shape}")

# ============================================================
# 4. TEXT-BASED FEATURES (N-GRAMS)
# ============================================================
print("Extracting character n-grams...")
char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=5000, min_df=5, max_df=0.95
)
char_ngrams_train = char_vectorizer.fit_transform(train_df["clean_text"])
char_ngrams_test = char_vectorizer.transform(test_df["clean_text"])
print(f"Character n-gram features: {char_ngrams_train.shape[1]}")

print("Extracting word TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=3000,
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    stop_words="english",
)
word_tfidf_train = tfidf_vectorizer.fit_transform(train_df["clean_text"])
word_tfidf_test = tfidf_vectorizer.transform(test_df["clean_text"])
print(f"Word TF-IDF features: {word_tfidf_train.shape[1]}")

# ============================================================
# 5. COMBINE ALL FEATURES (for possible classical model use)
# ============================================================
# NOTE: Scaling will be done per-fold inside CV loop to avoid data leakage.
# For now, combine raw features into sparse matrices (scaling done later).
X_train_content = hstack(
    [char_ngrams_train, word_tfidf_train, csr_matrix(train_stylo.values)]
)
X_test_content = hstack(
    [char_ngrams_test, word_tfidf_test, csr_matrix(test_stylo.values)]
)
print(f"Combined feature matrix shape: {X_train_content.shape}")

# ============================================================
# 6. TARGET ENCODING - Fit only inside CV loop to avoid leakage
# ============================================================
le = LabelEncoder()
# We'll transform inside the CV loop per fold; keep le for final submission mapping
le.fit(train_df["author"])
print(f"Class mapping: {dict(zip(range(3), le.classes_))}")

# ============================================================
# 7. CREATE TRAIN/VALIDATION SPLITS
# ============================================================
# Define y_train before using it
y_train = le.transform(train_df["author"].values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(skf.split(X_train_content, y_train))
val_idx = fold_indices[0][1]
train_idx = fold_indices[0][0]
print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")

# ============================================================
# 8. SAVE PREPROCESSED DATA (for backup)
# ============================================================
os.makedirs("./working", exist_ok=True)
save_npz("./working/X_train.npz", X_train_content[train_idx])
save_npz("./working/X_val.npz", X_train_content[val_idx])
save_npz("./working/X_test.npz", X_test_content)
np.save("./working/y_train.npy", y_train[train_idx])
np.save("./working/y_val.npy", y_train[val_idx])
save_npz("./working/X_train_full.npz", X_train_content)
np.save("./working/y_train_full.npy", y_train)
np.save("./working/test_ids.npy", test_df["id"].values)
np.save("./working/author_classes.npy", le.classes_)
train_df.to_pickle("./working/train_df.pkl")
test_df.to_pickle("./working/test_df.pkl")

with open("./working/char_vectorizer.pkl", "wb") as f:
    pickle.dump(char_vectorizer, f)
with open("./working/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
with open("./working/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Data processing complete. Files saved to ./working/")

# ============================================================
# 9. MODEL ARCHITECTURE DESIGN
# ============================================================
num_authors = len(le.classes_)
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================
# 9b. REGULARIZATION MODULES (Mixout + Stochastic Depth)
# ============================================================

class MixoutLinear(nn.Module):
    """Linear layer with Mixout regularization: stochastically replaces weights with pretrained values."""
    def __init__(self, original_linear, mixout_prob=0.2):
        super().__init__()
        self.mixout_prob = mixout_prob
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        # Keep a copy of the original pretrained weights and bias
        self.register_buffer('pretrained_weight', original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.register_buffer('pretrained_bias', original_linear.bias.data.clone())
        else:
            self.pretrained_bias = None
        # The actual trainable parameters
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, input):
        if self.training and self.mixout_prob > 0:
            # Create Bernoulli mask: 1 means use current weight, 0 means use pretrained weight
            mask = torch.bernoulli(
                torch.full_like(self.weight, 1.0 - self.mixout_prob)
            )
            # Mix current and pretrained weights
            weight = mask * self.weight + (1 - mask) * self.pretrained_weight
            if self.bias is not None:
                bias_mask = torch.bernoulli(
                    torch.full_like(self.bias, 1.0 - self.mixout_prob)
                )
                bias = bias_mask * self.bias + (1 - bias_mask) * self.pretrained_bias
            else:
                bias = None
        else:
            weight = self.weight
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


def apply_mixout_to_deberta(model, mixout_prob=0.2):
    """Replace all linear layers in DeBERTa's transformer with MixoutLinear wrappers."""
    for name, module in model.named_modules():
        # Only apply to transformer layers (not the classifier head)
        if "deberta.encoder.layer" in name and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for p in parent_name.split('.'):
                if p:
                    parent = getattr(parent, p) if hasattr(parent, p) else parent[int(p)] if p.isdigit() else parent[int(p)]
            # Replace the linear layer with MixoutLinear
            new_module = MixoutLinear(module, mixout_prob)
            if parent_name and parent_name.split('.')[-1].isdigit():
                # It's inside a list-like container (nn.ModuleList)
                idx = int(parent_name.split('.')[-1])
                parent[idx]._modules[child_name] = new_module
            else:
                setattr(parent, child_name, new_module)
    return model


def forward_with_stochastic_depth(model, input_ids, attention_mask, labels=None, survival_prob=0.9):
    """Custom forward pass with stochastic depth: randomly drop transformer layers during training."""
    if not model.training or survival_prob >= 1.0:
        # Inference or no dropping: use standard forward
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # Manual forward pass through encoder with stochastic depth
    config = model.config
    transformer = model.deberta
    hidden_states = transformer.embeddings(input_ids)

    # Create attention mask for DeBERTa format
    if attention_mask is not None:
        attention_mask = transformer.get_extended_attention_mask(attention_mask, input_ids.size(), input_ids.device)

    # Stochastic depth: each layer is kept with probability survival_prob
    num_layers = len(transformer.encoder.layer)
    keep_prob = torch.rand(num_layers, device=hidden_states.device)
    keep_mask = keep_prob < survival_prob

    # Ensure at least one layer is kept
    if not keep_mask.any():
        keep_mask[torch.randint(0, num_layers, (1,))] = True

    # Pass through each layer
    for i, layer_module in enumerate(transformer.encoder.layer):
        if keep_mask[i]:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

    # Pooling and classifier
    pooled_output = transformer.pooler(hidden_states) if hasattr(transformer, 'pooler') else hidden_states[:, 0, :]
    logits = model.classifier(pooled_output)

    loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

    return CustomOutput(loss=loss, logits=logits)


class CustomOutput:
    """Simple mock of transformers output to maintain interface compatibility."""
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


def get_layerwise_lr_params(model, base_lr=2e-5):
    """Group model parameters with different learning rates based on layer depth."""
    # DeBERTa-v3-small has 6 layers (indices 0-5) + embedding + classifier
    # Layer hierarchy: embeddings (deepest) -> layers 0,1,2,3,4,5 -> classifier (shallowest)
    param_groups = []

    # Embeddings layer - lowest LR
    embedding_params = []
    for name, param in model.deberta.embeddings.named_parameters():
        if param.requires_grad:
            embedding_params.append(param)
    param_groups.append({'params': embedding_params, 'lr': base_lr * 0.25, 'weight_decay': 0.01})

    # Split transformer layers into groups
    num_layers = len(model.deberta.encoder.layer)
    for layer_idx in range(num_layers):
        if layer_idx <= 1:  # Bottom layers
            lr_scale = 0.5
        elif layer_idx <= 3:  # Middle layers
            lr_scale = 1.0
        else:  # Top layers
            lr_scale = 1.5
        layer_params = []
        for name, param in model.deberta.encoder.layer[layer_idx].named_parameters():
            if param.requires_grad:
                layer_params.append(param)
        param_groups.append({'params': layer_params, 'lr': base_lr * lr_scale, 'weight_decay': 0.01})

    # Classifier head - highest LR
    classifier_params = []
    for name, param in model.classifier.named_parameters():
        if param.requires_grad:
            classifier_params.append(param)
    param_groups.append({'params': classifier_params, 'lr': base_lr * 2.5, 'weight_decay': 0.01})

    return param_groups

# ============================================================
# 10. PREPARE DATA FOR TRANSFORMER TRAINING
# ============================================================
print("Preparing data for transformer training...")
train_texts = train_df["clean_text"].values
train_labels = le.transform(train_df["author"].values)


def tokenize_texts(texts, max_length=512):
    return tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


print("Tokenizing training data...")
train_encodings = tokenize_texts(train_texts)
print("Tokenizing test data...")
test_texts = test_df["clean_text"].values
test_encodings = tokenize_texts(test_texts)

# ============================================================
# 11. TRAINING CONFIGURATION
# ============================================================
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 8
GRADIENT_CLIP_NORM = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 12. CROSS-VALIDATION TRAINING
# ============================================================
print(f"Starting 5-fold cross-validation training...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
all_test_probs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}/5")
    print(f"{'='*50}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_authors,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model.to(device)

    # Unfreeze all layers - fine-tune entire model
    for param in model.parameters():
        param.requires_grad = True

    # Use different learning rates: lower for pretrained backbone, higher for classifier head
    optimizer = AdamW(
        [
            {"params": model.deberta.parameters(), "lr": 2e-5, "weight_decay": 0.01},
            {"params": model.classifier.parameters(), "lr": 5e-5, "weight_decay": 0.01},
        ]
    )

    # Use fp32 training for stability (no autocast/scaler)
    scaler = None

    train_dataset = TensorDataset(
        train_encodings["input_ids"][train_idx],
        train_encodings["attention_mask"][train_idx],
        torch.tensor(train_labels[train_idx]),
    )
    val_dataset = TensorDataset(
        train_encodings["input_ids"][val_idx],
        train_encodings["attention_mask"][val_idx],
        torch.tensor(train_labels[val_idx]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 3
    best_val_log_loss = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / ACCUMULATION_STEPS

            loss.backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * ACCUMULATION_STEPS

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Check for NaN
                if torch.isnan(loss):
                    print("NaN detected in validation loss, skipping this batch")
                    continue

                total_val_loss += loss.item()
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                # Check for NaN in probs
                if torch.isnan(probs).any():
                    # Replace NaN with uniform distribution
                    probs = torch.ones_like(probs) / num_authors
                all_val_preds.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_true = np.concatenate(all_val_labels, axis=0)
        # Ensure no NaN in predictions
        min_prob = 1e-15
        val_preds = np.clip(val_preds, min_prob, 1 - min_prob)
        val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
        val_log_loss = log_loss(val_true, val_preds)

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Log Loss: {val_log_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/best_model_fold{fold}.pt")
            best_val_log_loss = val_log_loss
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f"./working/best_model_fold{fold}.pt"))
    fold_scores.append(best_val_log_loss)
    print(f"Fold {fold+1} best log loss: {best_val_log_loss:.4f}")

    model.eval()
    test_probs = []

    with torch.no_grad():
        test_dataset = TensorDataset(
            test_encodings["input_ids"], test_encodings["attention_mask"]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        for batch in test_loader:
            input_ids, attention_mask = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            # Check for NaN
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / num_authors
            test_probs.append(probs.cpu().numpy())

    fold_test_probs = np.concatenate(test_probs, axis=0)
    all_test_probs.append(fold_test_probs)

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 13. FINAL INFERENCE AND SUBMISSION
# ============================================================
print("\nPerforming final test inference...")
final_test_probs = np.mean(all_test_probs, axis=0)

eps = 1e-15
final_test_probs = np.clip(final_test_probs, eps, 1 - eps)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

print("Creating submission file...")
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        le.classes_[0]: final_test_probs[:, 0],
        le.classes_[1]: final_test_probs[:, 1],
        le.classes_[2]: final_test_probs[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_d7977a43624649708842153fa07ace1a.csv", index=False)

# ============================================================
# 14. FINAL VALIDATION SCORE
# ============================================================
final_val_score = np.mean(fold_scores)
print(f"Cross-validation log-loss scores: {fold_scores}")
print(f"Average CV log-loss: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
print(f"Final Validation Score: {final_val_score}")