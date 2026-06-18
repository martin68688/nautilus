import os
os.sched_setaffinity(0, {46, 47, 48, 49, 50})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet
import nltk
import random
import warnings

warnings.filterwarnings("ignore")

# Download NLTK data for WordNet synonym replacement
nltk.download('wordnet', quiet=True)

# ==================== DATA LOADING ====================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode labels
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_authors = len(label_encoder.classes_)
author_names = label_encoder.classes_
print(f"Authors: {author_names}")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

train_texts = train_df["text"].tolist()
train_labels = train_df["author_encoded"].values
test_texts = test_df["text"].tolist()
test_ids = test_df["id"].values

# ==================== STATISTICAL FEATURES ====================
def extract_statistical_features(text):
    import string
    text_len = len(text)
    word_count = len(text.split())
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    capitalization_count = sum(1 for c in text if c.isupper())
    stop_words = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because',
                       'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                       'to', 'in', 'on', 'that', 'this', 'it', 'and', 'not', 'so', 'will'])
    text_lower = text.lower().split()
    stopword_count = sum(1 for w in text_lower if w in stop_words)
    punct_density = punctuation_count / max(text_len, 1)
    cap_ratio = capitalization_count / max(text_len, 1)
    stopword_ratio = stopword_count / max(word_count, 1)
    return [text_len, word_count, punct_density, cap_ratio, stopword_ratio]

# Extract features for all texts
all_texts = train_texts + test_texts
all_feature_list = [extract_statistical_features(t) for t in all_texts]
feature_array = np.array(all_feature_list, dtype=np.float32)
# Normalize per split - FIT ONLY ON TRAIN to avoid data leakage
train_feature_array = feature_array[:len(train_texts)]
test_feature_array = feature_array[len(train_texts):]
scaler_feat = StandardScaler()
train_feature_array_norm = scaler_feat.fit_transform(train_feature_array)
test_feature_array_norm = scaler_feat.transform(test_feature_array)
# Prepend as special feature tokens: convert normalized features to tokens
FEATURE_VOCAB_SIZE = 100
def features_to_token_ids(features_norm):
    # Scale normalized features to [0, FEATURE_VOCAB_SIZE-1] and cast to int
    scaled = np.clip(np.floor((features_norm + 3) / 6 * (FEATURE_VOCAB_SIZE - 1)), 0, FEATURE_VOCAB_SIZE - 1).astype(int)
    return scaled.tolist()

train_feature_tokens_list = [features_to_token_ids(f) for f in train_feature_array_norm]
test_feature_tokens_list = [features_to_token_ids(f) for f in test_feature_array_norm]

# ==================== MODEL CONFIGURATION ====================
# Define ensemble configurations
model_configs = [
    {"name": "roberta-large", "model_name": "roberta-large", "lr": 2e-5, "aug_type": "token_deletion"},
    {"name": "longformer", "model_name": "allenai/longformer-base-4096", "lr": 1e-5, "aug_type": "synonym_replacement"},
    {"name": "xlm-roberta-large", "model_name": "xlm-roberta-large", "lr": 2e-5, "aug_type": "mask_insertion"},
]

max_length = 256
batch_size = 8  # Per model effective batch size 16 via gradient accumulation
effective_batch_size = 16
accumulation_steps = effective_batch_size // batch_size
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Gradient accumulation steps: {accumulation_steps}")

# Prepare tokenizers and models
ensemble_tokenizers = {}
ensemble_models = {}
for config in model_configs:
    print(f"Loading {config['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    ensemble_tokenizers[config["name"]] = tokenizer


# ==================== TOKEN-LEVEL AUGMENTATIONS ====================
def token_deletion_augment(text, p=0.03):
    tokens = text.split()
    if len(tokens) < 5:
        return text
    to_keep = [t for t in tokens if random.random() > p]
    return " ".join(to_keep) if to_keep else text

def synonym_replacement_augment(text, p=0.02):
    tokens = text.split()
    if len(tokens) < 5:
        return text
    new_tokens = []
    for t in tokens:
        if random.random() < p:
            synsets = wordnet.synsets(t)
            if synsets:
                lemmas = [l for s in synsets for l in s.lemmas()]
                if lemmas:
                    synonym = random.choice(lemmas).name().replace('_', ' ')
                    new_tokens.append(synonym)
                    continue
        new_tokens.append(t)
    return " ".join(new_tokens)

def mask_insertion_augment(text, p=0.02):
    tokens = text.split()
    if len(tokens) < 5:
        return text
    new_tokens = []
    for t in tokens:
        if random.random() < p:
            new_tokens.append("[MASK]")
        new_tokens.append(t)
    return " ".join(new_tokens)

augmentation_functions = {
    "token_deletion": token_deletion_augment,
    "synonym_replacement": synonym_replacement_augment,
    "mask_insertion": mask_insertion_augment,
}

# ==================== DATASET CLASS ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer_name=None, feature_tokens_list=None, augment=False, aug_type=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer_name = tokenizer_name
        self.tokenizer = ensemble_tokenizers.get(tokenizer_name)
        self.feature_tokens_list = feature_tokens_list
        self.augment = augment
        self.aug_type = aug_type

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Apply augmentation during training
        if self.augment and self.aug_type is not None:
            aug_func = augmentation_functions.get(self.aug_type)
            if aug_func:
                text = aug_func(text)

        # Prepend feature tokens as special tokens
        if self.feature_tokens_list is not None:
            feature_tokens = self.feature_tokens_list[idx]
            # Convert feature tokens to text representation to prepend
            feature_text = " ".join([f"<FEAT_{ft}>" for ft in feature_tokens])
            text = feature_text + " " + text

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, attention_mask, label
        return input_ids, attention_mask


# ==================== TRAINING FUNCTION ====================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


# ==================== EVALUATION FUNCTION ====================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ==================== MAIN TRAINING LOOP (ENSEMBLE WITH 5-FOLD CV) ====================
print("Setting up stratified k-fold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
# Store OOF and test predictions per model per fold
all_oof_preds = {config["name"]: [] for config in model_configs}
all_test_preds = {config["name"]: [] for config in model_configs}
all_val_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n{'='*60}")
    print(f"===== Fold {fold+1}/5 =====")
    print(f"{'='*60}")

    fold_train_texts = [train_texts[i] for i in train_idx]
    fold_train_labels = train_labels[train_idx]
    fold_val_texts = [train_texts[i] for i in val_idx]
    fold_val_labels = train_labels[val_idx]

    # Store val labels for this fold
    all_val_labels.append(fold_val_labels)

    # For each model in the ensemble
    for config in model_configs:
        model_name = config["name"]
        model_hf_name = config["model_name"]
        lr = config["lr"]
        aug_type = config["aug_type"]
        tokenizer = ensemble_tokenizers[model_name]

        print(f"\n--- Training {model_name} on Fold {fold+1} ---")

        # Create datasets with augmentations and statistical features
        train_dataset = TextDataset(
            fold_train_texts, fold_train_labels,
            tokenizer_name=model_name,
            feature_tokens_list=[train_feature_tokens_list[i] for i in train_idx],
            augment=True, aug_type=aug_type
        )
        val_dataset = TextDataset(
            fold_val_texts, fold_val_labels,
            tokenizer_name=model_name,
            feature_tokens_list=[train_feature_tokens_list[i] for i in val_idx],
            augment=False
        )

        # Create test dataset for this model
        test_dataset = TextDataset(
            test_texts, labels=None,
            tokenizer_name=model_name,
            feature_tokens_list=test_feature_tokens_list,
            augment=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize model with dropout=0.3
        model = AutoModelForSequenceClassification.from_pretrained(
            model_hf_name,
            num_labels=num_authors,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
        )
        # Resize token embeddings if tokenizer has added special tokens
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs // accumulation_steps
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = GradScaler()

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler, device,
                accumulation_steps=accumulation_steps
            )

            val_preds, val_labels_epoch = evaluate(model, val_loader, device)

            # Clip probabilities
            epsilon = 1e-15
            val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
            val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(
                axis=1, keepdims=True
            )

            val_loss = log_loss(val_labels_epoch, val_preds_clipped)

            print(
                f"  {model_name} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Log Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"  Early stopping triggered for {model_name}")
                    break

        # Check if fold should be discarded (val loss > 0.35)
        if best_val_loss > 0.35:
            print(f"  Fold {fold+1} for {model_name} discarded (val_loss={best_val_loss:.4f} > 0.35)")
            # Add placeholder None to keep indexing
            all_oof_preds[model_name].append(None)
            all_test_preds[model_name].append(None)
            continue

        # Load best model and generate OOF predictions
        model.load_state_dict(best_model_state)
        oof_preds, _ = evaluate(model, val_loader, device)
        all_oof_preds[model_name].append(oof_preds)

        # Generate test predictions
        model.eval()
        model_test_preds = []
        with torch.no_grad():
            for input_ids, attention_mask in test_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                model_test_preds.append(probs.cpu().numpy())
        model_test_preds = np.concatenate(model_test_preds, axis=0)
        all_test_preds[model_name].append(model_test_preds)

        # Clean up to save memory
        model.cpu()
        del model
        torch.cuda.empty_cache()

    val_scores.append(best_val_loss)
    print(f"Fold {fold+1} best validation log loss: {best_val_loss:.4f}")

print(f"\nAverage validation log loss: {np.mean(val_scores):.4f}")

# ==================== META-LEARNER TRAINING ====================
print("\n" + "="*60)
print("Training meta-learner (Logistic Regression)...")
print("="*60)

# Build feature matrix from OOF predictions
feature_list = []
label_list = []
for fold_idx in range(5):
    fold_oofs = []
    for config in model_configs:
        model_name = config["name"]
        oof_fold = all_oof_preds[model_name][fold_idx]
        if oof_fold is not None:
            fold_oofs.append(oof_fold)
    if len(fold_oofs) > 0 and fold_idx < len(all_val_labels):
        # Concatenate all model predictions for this fold
        concat_oof = np.concatenate(fold_oofs, axis=1)  # shape: (n_val, num_models * 3)
        feature_list.append(concat_oof)
        label_list.append(all_val_labels[fold_idx])

if len(feature_list) == 0:
    print("WARNING: No valid folds retained for meta-learner. Falling back to simple averaging.")
    # Fallback: simple average of all test predictions
    final_test_preds = np.zeros((len(test_ids), num_authors))
    count = 0
    for config in model_configs:
        model_name = config["name"]
        for fold_idx in range(5):
            if all_test_preds[model_name][fold_idx] is not None:
                final_test_preds += all_test_preds[model_name][fold_idx]
                count += 1
    if count > 0:
        final_test_preds /= count
    else:
        print("ERROR: No test predictions available!")
        final_test_preds = np.ones((len(test_ids), num_authors)) / num_authors
else:
    X_meta = np.concatenate(feature_list, axis=0)
    y_meta = np.concatenate(label_list, axis=0)

    print(f"Meta feature shape: {X_meta.shape}")
    print(f"Meta labels shape: {y_meta.shape}")

    # Train Logistic Regression meta-classifier
    meta_clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, multi_class='multinomial')
    meta_clf.fit(X_meta, y_meta)

    # Build test feature matrix and predict
    test_feature_list = []
    for fold_idx in range(5):
        fold_test_preds = []
        for config in model_configs:
            model_name = config["name"]
            test_pred_fold = all_test_preds[model_name][fold_idx]
            if test_pred_fold is not None:
                fold_test_preds.append(test_pred_fold)
        if len(fold_test_preds) > 0:
            concat_test = np.concatenate(fold_test_preds, axis=1)
            test_feature_list.append(concat_test)

    if len(test_feature_list) > 0:
        X_test_meta = np.mean(test_feature_list, axis=0)  # Average across folds
        final_test_preds = meta_clf.predict_proba(X_test_meta)
    else:
        print("ERROR: No test predictions for meta-learner, falling back.")
        final_test_preds = np.ones((len(test_ids), num_authors)) / num_authors

# Clip and normalize final predictions
epsilon = 1e-15
final_test_preds = np.clip(final_test_preds, epsilon, 1 - epsilon)
final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

# ==================== SUBMISSION ====================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": final_test_preds[:, 0],
        "HPL": final_test_preds[:, 1],
        "MWS": final_test_preds[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_8cbb24e89f6c4d6d9632206b14b5d8cb.csv", index=False)
print(f"Submission saved to ./submission/submission_8cbb24e89f6c4d6d9632206b14b5d8cb.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

final_val_score = np.mean(val_scores) if len(val_scores) > 0 else 0.0
print(f"Final Validation Score: {final_val_score:.4f}")
print("Ensemble training complete!")