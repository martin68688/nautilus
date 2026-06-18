import os
os.sched_setaffinity(0, {154, 156, 61, 62, 63})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os, gc, sys, math, json, random, warnings, pickle, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
ROOT = "./input/"
OUT = "./submission/"
os.makedirs(OUT, exist_ok=True)
os.makedirs("./working/", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
MAX_LEN = 256
BATCH_SIZE = 8
ACCUM_STEPS = 4
EPOCHS = 5
LR = 2e-5
MODEL_NAME = "microsoft/deberta-v3-large"
NUM_LABELS = 3
CLASS_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}
INV_CLASS = {0: "EAP", 1: "HPL", 2: "MWS"}

train_df = pd.read_csv(ROOT + "train.csv")
test_df = pd.read_csv(ROOT + "test.csv")
sample_sub = pd.read_csv(ROOT + "sample_submission.csv")

train_df["label"] = train_df["author"].map(CLASS_MAP)
all_texts = pd.concat([train_df["text"], test_df["text"]]).values


# ---- Stylometric Feature Engineering ----
def get_stylometric_features(texts):
    features = []
    for text in texts:
        words = str(text).split()
        sentences = str(text).split(".")
        chars = len(str(text))
        word_len = len(words)
        sent_len = len(sentences)
        avg_word_len = chars / max(word_len, 1)
        avg_sent_len = word_len / max(sent_len, 1)
        punct_count = sum(1 for c in str(text) if c in ".,;:!?'\"-")
        comma_ratio = str(text).count(",") / max(word_len, 1)
        exclam_ratio = str(text).count("!") / max(word_len, 1)
        quest_ratio = str(text).count("?") / max(word_len, 1)
        colon_ratio = str(text).count(":") / max(word_len, 1)
        semicolon_ratio = str(text).count(";") / max(word_len, 1)
        quote_ratio = str(text).count('"') / max(word_len, 1)
        dash_ratio = str(text).count("—") / max(word_len, 1)
        cap_ratio = sum(1 for c in str(text) if c.isupper()) / max(chars, 1)
        digit_ratio = sum(1 for c in str(text) if c.isdigit()) / max(chars, 1)
        unique_word_ratio = len(set(w.lower() for w in words)) / max(word_len, 1)
        features.append(
            [
                avg_word_len,
                avg_sent_len,
                punct_count / max(chars, 1),
                comma_ratio,
                exclam_ratio,
                quest_ratio,
                colon_ratio,
                semicolon_ratio,
                quote_ratio,
                dash_ratio,
                cap_ratio,
                digit_ratio,
                unique_word_ratio,
                word_len,
                chars,
            ]
        )
    return np.array(features)


# Compute stylometric features per fold inside the CV loop to avoid leakage
# Will compute per fold in the XGBoost section


class SpookyDataset(Dataset):
    def __init__(
        self,
        texts,
        labels=None,
        tokenizer=None,
        max_len=MAX_LEN,
        trunc_strategy="first",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.trunc_strategy = trunc_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.trunc_strategy == "last" and len(text) > self.max_len * 4:
            text = text[-self.max_len * 4 :]
        elif self.trunc_strategy == "random" and len(text) > self.max_len * 4:
            start = random.randint(0, len(text) - self.max_len * 4)
            text = text[start : start + self.max_len * 4]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class DebertaForSequenceClassificationWithPooling(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.config.hidden_dropout_prob = 0.3
        self.attention_pool = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512), nn.Tanh(), nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )
        self.label_smoothing = 0.1
        self._init_weights()
        self.frozen = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        """Freeze the encoder parameters, train only head and attention pooling."""
        for param in self.deberta.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.deberta.parameters():
            param.requires_grad = True
        self.frozen = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        attn_weights = self.attention_pool(hidden)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (attn_weights * hidden).sum(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        if labels is not None:
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            return loss, logits, pooled
        return logits, pooled


def stylistic_augment(text):
    """Apply stylistic augmentation to text: function word substitution,
    archaic word insertion, and punctuation perturbation."""
    words = str(text).split()
    function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'shall', 'should', 'may', 'might', 'must', 'can',
                     'could', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'out', 'off', 'over', 'under',
                     'again', 'further', 'then', 'once', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                     'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                     'because', 'and', 'but', 'or', 'if', 'while', 'that', 'this',
                     'these', 'those', 'it', 'its', 'they', 'them', 'their', 'what',
                     'which', 'who', 'whom', 'whose'}
    archaic_words = ['hath', 'doth', 'thou', 'thee', 'thy', 'thine', 'whence',
                     'thence', 'hence', 'whither', 'thither', 'hither', 'perchance',
                     'forsooth', 'methinks', 'prithee', 'anon', 'ere', 'o\'er',
                     'ne\'er', 'oft', 'ofttimes', 'betwixt', 'twixt', 'unto',
                     'wilt', 'canst', 'dost', 'doth', 'art', 'wert', 'shalt',
                     'mayest', 'mightest', 'couldst', 'wouldst', 'shouldst',
                     'wherefore', 'therefor', 'hereto', 'thereto', 'whereto']

    augmented_words = words.copy()

    # Randomly substitute 20% of function words
    for i, word in enumerate(words):
        if word.lower() in function_words and random.random() < 0.2:
            alt_words = [w for w in function_words if w != word.lower()]
            if alt_words:
                replacement = random.choice(alt_words)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                augmented_words[i] = replacement

    # Insert archaic words with 10% chance
    new_words = augmented_words.copy()
    insert_count = 0
    for i in range(len(augmented_words)):
        if random.random() < 0.1 and i >= len(augmented_words) // 2:
            archaic = random.choice(archaic_words)
            new_words.insert(i + insert_count, archaic)
            insert_count += 1

    # Perturb punctuation with 15% chance
    result_words = new_words.copy()
    for i, word in enumerate(result_words):
        if random.random() < 0.15:
            if word.endswith('.'):
                result_words[i] = word[:-1] + random.choice(['.', '!', '?'])
            elif word.endswith(','):
                result_words[i] = word[:-1] + random.choice([',', ';', ':'])
            elif word.endswith('!'):
                result_words[i] = word[:-1] + random.choice(['!', '.', '?'])
            elif word.endswith('?'):
                result_words[i] = word[:-1] + random.choice(['?', '.', '!'])
            elif word.endswith(';'):
                result_words[i] = word[:-1] + random.choice([';', ',', ':'])
            elif word.endswith(':'):
                result_words[i] = word[:-1] + random.choice([':', ',', ';'])

    return ' '.join(result_words)


def custom_collate(batch):
    """Custom collate function that applies stylistic augmentation."""
    new_batch = []
    for item in batch:
        if 'labels' in item:
            text = item['text'] if 'text' in item else ''
            if random.random() < 0.5:
                text = stylistic_augment(text)
            # Re-tokenize with augmentation
            enc = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            new_item = {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels': item['labels']
            }
            new_batch.append(new_item)
        else:
            new_batch.append(item)

    # Pad and stack
    if new_batch:
        input_ids = torch.stack([b['input_ids'] for b in new_batch])
        attention_mask = torch.stack([b['attention_mask'] for b in new_batch])
        result = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'labels' in new_batch[0]:
            result['labels'] = torch.stack([b['labels'] for b in new_batch])
        return result
    return batch


class AugmentedSpookyDataset(SpookyDataset):
    """Dataset that stores original text for augmentation."""
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        item = {'text': text}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(model, loader, optimizer, scaler, scheduler=None, accumulate=1, stage='unfrozen'):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        if DEVICE.type == "cuda":
            with autocast():
                loss, logits, _ = model(input_ids, attention_mask, labels)
                loss = loss / accumulate
                loss = torch.clamp(loss, max=10.0)
            scaler.scale(loss).backward()
            if (i + 1) % accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
        else:
            loss, logits, _ = model(input_ids, attention_mask, labels)
            loss = loss / accumulate
            loss = torch.clamp(loss, max=10.0)
            loss.backward()
            if (i + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
        total_loss += loss.item() * accumulate
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with autocast():
                loss, logits, _ = model(input_ids, attention_mask, labels)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    l = log_loss(all_labels, all_preds)
    return l, all_preds


def predict_test(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                logits, _ = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds)


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_oof = np.zeros((len(train_df), NUM_LABELS))
all_test_preds = np.zeros((len(test_df), NUM_LABELS))
val_ll_list = []

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train_df["text"], train_df["label"])
):
    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
    trn_texts = train_df["text"].iloc[trn_idx].values
    val_texts = train_df["text"].iloc[val_idx].values
    trn_labels = train_df["label"].iloc[trn_idx].values
    val_labels = train_df["label"].iloc[val_idx].values

    trn_ds = SpookyDataset(trn_texts, trn_labels, tokenizer, trunc_strategy="first")
    val_ds = SpookyDataset(val_texts, val_labels, tokenizer, trunc_strategy="first")
    test_ds = SpookyDataset(
        test_df["text"].values, None, tokenizer, trunc_strategy="first"
    )

    trn_loader = DataLoader(
        trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(
        DEVICE
    )

    # Create augmented training dataset and loader
    aug_trn_ds = AugmentedSpookyDataset(trn_texts, trn_labels, tokenizer, trunc_strategy="first")
    trn_loader_aug = DataLoader(
        aug_trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=custom_collate
    )

    # Two-stage training with differential learning rates
    EPOCHS_STAGE1 = 2  # frozen encoder
    EPOCHS_STAGE2 = 3  # unfrozen encoder

    # Stage 1: Freeze encoder, train only head and attention pooling
    model.freeze_encoder()

    # Optimizer for stage 1: only head and attention pooling parameters
    head_params = list(model.classifier.parameters()) + list(model.attention_pool.parameters()) + [model.dropout.weight] if hasattr(model.dropout, 'weight') else list(model.classifier.parameters()) + list(model.attention_pool.parameters())
    optimizer_stage1 = AdamW(head_params, lr=2e-5, weight_decay=0.1)
    total_steps_stage1 = len(trn_loader_aug) // ACCUM_STEPS * EPOCHS_STAGE1
    scheduler_stage1 = transformers.get_linear_schedule_with_warmup(
        optimizer_stage1,
        num_warmup_steps=int(0.1 * total_steps_stage1),
        num_training_steps=total_steps_stage1,
    )
    scaler = GradScaler()

    print("Stage 1: Frozen encoder, training head only...")
    for epoch in range(EPOCHS_STAGE1):
        train_loss = train_epoch(
            model, trn_loader_aug, optimizer_stage1, scaler, scheduler_stage1, ACCUM_STEPS, stage='frozen'
        )
        val_ll, _ = evaluate(model, val_loader)
        print(
            f"Stage 1 - Epoch {epoch+1}/{EPOCHS_STAGE1} - Train Loss: {train_loss:.4f}, Val LL: {val_ll:.4f}"
        )

    # Stage 2: Unfreeze encoder, train all parameters with differential LR
    model.unfreeze_encoder()

    # Differential learning rates: backbone=1e-5, head=2e-5
    optimizer_grouped_parameters = [
        {'params': model.deberta.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 2e-5},
        {'params': model.attention_pool.parameters(), 'lr': 2e-5},
        {'params': model.dropout.parameters(), 'lr': 2e-5},
    ]
    optimizer_stage2 = AdamW(optimizer_grouped_parameters, lr=1e-5, weight_decay=0.1)
    total_steps_stage2 = len(trn_loader_aug) // ACCUM_STEPS * EPOCHS_STAGE2
    scheduler_stage2 = transformers.get_linear_schedule_with_warmup(
        optimizer_stage2,
        num_warmup_steps=int(0.1 * total_steps_stage2),
        num_training_steps=total_steps_stage2,
    )

    print("Stage 2: Unfrozen encoder, full fine-tuning...")
    best_ll = float("inf")
    best_model_state = None
    patience = 3
    no_improve = 0

    # For SWA: store weights from last 3 epochs
    swa_weights = []

    for epoch in range(EPOCHS_STAGE2):
        train_loss = train_epoch(
            model, trn_loader_aug, optimizer_stage2, scaler, scheduler_stage2, ACCUM_STEPS, stage='unfrozen'
        )
        val_ll, _ = evaluate(model, val_loader)
        print(
            f"Stage 2 - Epoch {epoch+1}/{EPOCHS_STAGE2} - Train Loss: {train_loss:.4f}, Val LL: {val_ll:.4f}"
        )

        # Store weights for SWA (last 3 epochs)
        swa_weights.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if len(swa_weights) > 3:
            swa_weights.pop(0)

        if val_ll < best_ll:
            best_ll = val_ll
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
            torch.save(best_model_state, f"./working/deberta_fold{fold}.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Apply SWA: average weights from last 3 epochs
    if len(swa_weights) >= 2:
        print("Applying SWA averaging...")
        swa_state_dict = {}
        for key in swa_weights[0].keys():
            swa_state_dict[key] = torch.stack([w[key] for w in swa_weights]).mean(dim=0)
        model.load_state_dict(swa_state_dict)

    # Use best model if SWA doesn't improve
    model.load_state_dict(torch.load(f"./working/deberta_fold{fold}.pt"))
    model.to(DEVICE)
    _, val_probs = evaluate(model, val_loader)
    test_probs = predict_test(model, test_loader)

    all_oof[val_idx] = val_probs
    all_test_preds += test_probs / N_FOLDS
    val_ll_list.append(best_ll)
    print(f"Fold {fold+1} best LL: {best_ll:.4f}")
    del model, trn_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

print(f'\nDeBERTa CV OOF LL: {log_loss(train_df["label"], all_oof):.4f}')


# ---- XGBoost with stylometric + DeBERTa embeddings ----
# Extract pooled embeddings from DeBERTa for meta-features
def extract_embeddings(model, loader):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                _, pooled = model(input_ids, attention_mask)
            all_embs.append(pooled.cpu().numpy())
    return np.concatenate(all_embs)


# Re-load a model to extract embeddings for meta-features
meta_model = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(
    DEVICE
)
meta_model.load_state_dict(torch.load(f"./working/deberta_fold0.pt"))
meta_model.to(DEVICE)
meta_model.eval()

full_train_ds = SpookyDataset(
    train_df["text"].values, None, tokenizer, trunc_strategy="first"
)
full_train_loader = DataLoader(
    full_train_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2
)
train_embs = extract_embeddings(meta_model, full_train_loader)

full_test_ds = SpookyDataset(
    test_df["text"].values, None, tokenizer, trunc_strategy="first"
)
full_test_loader = DataLoader(
    full_test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2
)
test_embs = extract_embeddings(meta_model, full_test_loader)

del meta_model
gc.collect()
torch.cuda.empty_cache()

# Re-extract embeddings in a proper cross-validated manner
# First, get fold-wise embeddings and stylometric features
skf_xgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
xgb_oof = np.zeros((len(train_df), NUM_LABELS))
xgb_test = np.zeros((len(test_df), NUM_LABELS))
xgb_scores = []

for fold, (trn_idx, val_idx) in enumerate(skf_xgb.split(train_df["text"].values, train_df["label"].values)):
    print(f"XGB Fold {fold+1}/5")

    # Compute stylometric features separately per fold
    trn_texts_fold = train_df["text"].iloc[trn_idx].values
    val_texts_fold = train_df["text"].iloc[val_idx].values
    stylo_trn = get_stylometric_features(trn_texts_fold)
    stylo_val = get_stylometric_features(val_texts_fold)

    # Load corresponding fold's DeBERTa model for embeddings
    meta_model_fold = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(DEVICE)
    meta_model_fold.load_state_dict(torch.load(f"./working/deberta_fold{fold}.pt"))
    meta_model_fold.to(DEVICE)
    meta_model_fold.eval()

    # Extract embeddings for train fold
    trn_ds_fold = SpookyDataset(trn_texts_fold, None, tokenizer, trunc_strategy="first")
    trn_loader_fold = DataLoader(trn_ds_fold, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    trn_embs_fold = extract_embeddings(meta_model_fold, trn_loader_fold)

    # Extract embeddings for validation fold
    val_ds_fold = SpookyDataset(val_texts_fold, None, tokenizer, trunc_strategy="first")
    val_loader_fold = DataLoader(val_ds_fold, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    val_embs_fold = extract_embeddings(meta_model_fold, val_loader_fold)

    # Extract embeddings for test (use same model)
    test_ds_fold = SpookyDataset(test_df["text"].values, None, tokenizer, trunc_strategy="first")
    test_loader_fold = DataLoader(test_ds_fold, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    test_embs_fold = extract_embeddings(meta_model_fold, test_loader_fold)

    del meta_model_fold
    gc.collect()
    torch.cuda.empty_cache()

    # Combine features
    X_tr = np.concatenate([stylo_trn, trn_embs_fold], axis=1)
    X_val = np.concatenate([stylo_val, val_embs_fold], axis=1)
    y_tr = train_df["label"].iloc[trn_idx].values
    y_val = train_df["label"].iloc[val_idx].values

    # Enhanced XGBoost with more regularization
    xgb_model_fold = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        num_class=NUM_LABELS,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=0,
    )

    eval_set = [(X_val, y_val)]
    xgb_model_fold.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
    val_probs = xgb_model_fold.predict_proba(X_val)
    xgb_oof[val_idx] = val_probs
    xgb_test += xgb_model_fold.predict_proba(
        np.concatenate([get_stylometric_features(test_df["text"].values), test_embs_fold], axis=1)
    ) / 5
    score = log_loss(y_val, val_probs)
    xgb_scores.append(score)
    print(f"XGB Fold {fold+1} LL: {score:.4f}")

print(f"XGB CV OOF LL: {log_loss(train_df['label'].values, xgb_oof):.4f}")

# ---- Logistic Regression on embeddings only (also needs proper CV) ----
# Re-extract embeddings per fold - reuse the fold loop above
# But we already have fold indices, so we can re-run
lr_oof = np.zeros((len(train_df), NUM_LABELS))
lr_test = np.zeros((len(test_df), NUM_LABELS))

for fold, (trn_idx, val_idx) in enumerate(skf_xgb.split(train_df["text"].values, train_df["label"].values)):
    print(f"LR Fold {fold+1}/5")

    # Load corresponding fold's DeBERTa model
    lr_model_fold = DebertaForSequenceClassificationWithPooling(MODEL_NAME, NUM_LABELS).to(DEVICE)
    lr_model_fold.load_state_dict(torch.load(f"./working/deberta_fold{fold}.pt"))
    lr_model_fold.to(DEVICE)
    lr_model_fold.eval()

    trn_texts_lr = train_df["text"].iloc[trn_idx].values
    val_texts_lr = train_df["text"].iloc[val_idx].values

    trn_ds_lr = SpookyDataset(trn_texts_lr, None, tokenizer, trunc_strategy="first")
    trn_loader_lr = DataLoader(trn_ds_lr, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    trn_embs_lr = extract_embeddings(lr_model_fold, trn_loader_lr)

    val_ds_lr = SpookyDataset(val_texts_lr, None, tokenizer, trunc_strategy="first")
    val_loader_lr = DataLoader(val_ds_lr, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    val_embs_lr = extract_embeddings(lr_model_fold, val_loader_lr)

    test_ds_lr = SpookyDataset(test_df["text"].values, None, tokenizer, trunc_strategy="first")
    test_loader_lr = DataLoader(test_ds_lr, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    test_embs_lr = extract_embeddings(lr_model_fold, test_loader_lr)

    del lr_model_fold
    gc.collect()
    torch.cuda.empty_cache()

    lr = LogisticRegression(C=1.0, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=SEED)
    lr.fit(trn_embs_lr, train_df["label"].iloc[trn_idx].values)
    val_probs = lr.predict_proba(val_embs_lr)
    lr_oof[val_idx] = val_probs
    lr_test += lr.predict_proba(test_embs_lr) / 5

print(f"LR CV OOF LL: {log_loss(train_df['label'].values, lr_oof):.4f}")

# ---- Learned Weighted Ensemble via Logistic Regression blender ----
# Use meta-learner to find optimal weights
stacked_train = np.stack([all_oof, xgb_oof, lr_oof], axis=-1)  # (N, 3, 3)
stacked_test = np.stack([all_test_preds, xgb_test, lr_test], axis=-1)  # (N_test, 3, 3)

# Flatten: each model's predictions become features
stacked_train_flat = stacked_train.reshape(len(train_df), -1)  # (N, 9)
stacked_test_flat = stacked_test.reshape(len(test_df), -1)

blender = LogisticRegression(
    C=0.1, multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=SEED
)
blender_ll = 0

for fold, (trn_idx, val_idx) in enumerate(
    skf_xgb.split(stacked_train_flat, train_df["label"].values)
):
    X_tr, X_val = stacked_train_flat[trn_idx], stacked_train_flat[val_idx]
    y_tr, y_val = train_df["label"].values[trn_idx], train_df["label"].values[val_idx]
    blender.fit(X_tr, y_tr)
    val_probs = blender.predict_proba(X_val)
    blender_ll += log_loss(y_val, val_probs)

blender_ll /= 5
print(f"Blender CV OOF LL: {blender_ll:.4f}")

# Final blending
blender.fit(stacked_train_flat, train_df["label"].values)
final_preds = blender.predict_proba(stacked_test_flat)

final_validation_score = log_loss(
    train_df["label"].values, blender.predict_proba(stacked_train_flat)
)
print(f"Final Validation Score: {final_validation_score:.4f}")

# Generate submission
sub = pd.DataFrame({"id": test_df["id"].values})
for j, name in enumerate(["EAP", "HPL", "MWS"]):
    sub[name] = final_preds[:, j]
sub.to_csv(OUT + "submission_a610b2ac73ee419d952f00d1b7d3746a.csv", index=False)

# Cleanup
for f in os.listdir("./working/"):
    if f.endswith(".pt"):
        os.remove(os.path.join("./working/", f))

print("Submission saved successfully!")