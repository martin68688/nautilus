import os
os.sched_setaffinity(0, {4, 5, 6, 7, 8, 9, 12, 13, 14})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold, same split as original)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        # Multi-Head Attention pooling layer
        self.mha_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True,
        )
        # Projection layer to combine pooled representations
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.head = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        # Multi-Head Attention pooling: use learned query (mean of all tokens) attending to all tokens
        query = hidden_states.mean(dim=1, keepdim=True)  # (batch, 1, hidden_size)
        attn_output, _ = self.mha_pool(query, hidden_states, hidden_states, key_padding_mask=~attention_mask.bool())
        # Squeeze the query dimension
        mha_pooled = attn_output.squeeze(1)  # (batch, hidden_size)
        # Mean pooling across sequence length
        mean_pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        # Max pooling across sequence length
        max_pooled, _ = hidden_states.max(dim=1)  # (batch, hidden_size)
        # Concatenate all three representations
        pooled = torch.cat([mha_pooled, mean_pooled, max_pooled], dim=1)  # (batch, hidden_size * 3)
        # Project to hidden_size with dropout
        pooled = self.pool_proj(pooled)
        logits = self.head(pooled)
        return logits


model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.4)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Collect backbone unfrozen params (last 8 layers)
backbone_unfrozen_params = []
for layer in model.backbone.deberta.encoder.layer[-8:]:
    for name, param in layer.named_parameters():
        if "bias" not in name and "LayerNorm" not in name:
            backbone_unfrozen_params.append(param)

# Collect head params (only the new single linear layer)
head_params = list(model.head.parameters())

optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 2e-5,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.05, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_unfrozen_params):,}")
print(f"Head params: {sum(p.numel() for p in head_params):,}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# BACK-TRANSLATION AUGMENTATION
# ============================================================
from transformers import MarianMTModel, MarianTokenizer

# Offline pre-compute back-translated versions (English -> French, English -> German, English -> Spanish)
# Using Helsinki-NLP OpusMT models
backtranslation_models = {
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
    "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    "es": ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),
}

# Pre-compute back-translations for all training texts
def generate_backtranslations(texts, device=device):
    bt_texts = {lang: [] for lang in backtranslation_models}
    for lang, (to_lang_model, back_to_en_model) in backtranslation_models.items():
        print(f"Generating back-translations via {lang}...")
        tokenizer_to = MarianTokenizer.from_pretrained(to_lang_model)
        model_to = MarianMTModel.from_pretrained(to_lang_model).to(device)
        tokenizer_back = MarianTokenizer.from_pretrained(back_to_en_model)
        model_back = MarianMTModel.from_pretrained(back_to_en_model).to(device)

        batch_size_bt = 16
        for i in range(0, len(texts), batch_size_bt):
            batch_texts = texts[i:i+batch_size_bt].tolist()

            # Translate to intermediate language
            encoded_to = tokenizer_to(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                translated_ids = model_to.generate(**encoded_to, max_length=512)
            translated_texts = tokenizer_to.batch_decode(translated_ids, skip_special_tokens=True)

            # Translate back to English
            encoded_back = tokenizer_back(translated_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                back_translated_ids = model_back.generate(**encoded_back, max_length=512)
            back_translated_texts = tokenizer_back.batch_decode(back_translated_ids, skip_special_tokens=True)

            bt_texts[lang].extend(back_translated_texts)

        # Clean up to save memory
        del model_to, tokenizer_to, model_back, tokenizer_back
        torch.cuda.empty_cache()
        print(f"  Done: {len(bt_texts[lang])} texts")

    return bt_texts

# Get original texts for training (before split to compute back-translations only on train set)
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Only compute back-translations on training portion (not validation)
train_texts_for_bt = train_texts_orig[train_idx]
print("Pre-computing back-translations for training texts...")
bt_texts = generate_backtranslations(train_texts_for_bt, device)
print("Back-translations complete.")


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, bt_texts=None, bt_prob=0.15):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bt_texts = bt_texts
        self.bt_prob = bt_prob
        self.bt_langs = list(bt_texts.keys()) if bt_texts is not None else []

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Apply back-translation augmentation with probability bt_prob
        if self.bt_texts is not None and np.random.random() < self.bt_prob and len(self.bt_langs) > 0:
            lang = np.random.choice(self.bt_langs)
            if idx < len(self.bt_texts[lang]):
                text = self.bt_texts[lang][idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Use previously computed indices for train/validation split
train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length, bt_texts=bt_texts, bt_prob=0.15
)
val_dataset = SpookyDataset(val_texts_final, val_labels_final, tokenizer, max_length)
test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
num_epochs = 30
patience = 5
best_val_score = float("inf")
epochs_no_improve = 0
scaler_grad = GradScaler()

# OneCycleLR scheduler - step per epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[2e-5, 5e-5],
    steps_per_epoch=1,
    epochs=num_epochs,
    pct_start=0.1,
    final_div_factor=1000,
)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        total_train_loss += loss.item()
        num_train_batches += 1

    # Step scheduler per epoch
    scheduler.step()

    avg_train_loss = total_train_loss / num_train_batches

    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

            total_val_loss += loss.item()
            num_val_batches += 1
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_probs = np.concatenate(all_val_probs, axis=0)
    val_true = np.concatenate(all_val_labels, axis=0)

    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

    val_score = log_loss(val_true, val_probs_clipped)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f} | LR: {current_lr:.2e}"
    )

    if val_score < best_val_score:
        best_val_score = val_score
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./working/best_model_4cd8f67beff64d92a740688e38cd5073.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model_4cd8f67beff64d92a740688e38cd5073.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        all_val_probs.append(probs.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_probs = np.concatenate(all_val_probs, axis=0)
val_true = np.concatenate(all_val_labels, axis=0)
val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
final_val_score = log_loss(val_true, val_probs_clipped)

# ============================================================
# PSEUDO-LABELING PHASE
# ============================================================
print("Starting pseudo-labeling phase...")

# Generate pseudo-labels for test set with confidence threshold
model.eval()
all_test_probs_pl = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        all_test_probs_pl.append(probs.cpu().numpy())

test_probs_pl = np.concatenate(all_test_probs_pl, axis=0)
test_confidence = np.max(test_probs_pl, axis=1)
high_conf_mask = test_confidence >= 0.95

# Combine high-confidence test samples with training data
pseudo_texts = test_texts[high_conf_mask]
pseudo_labels = np.argmax(test_probs_pl[high_conf_mask], axis=1)

augmented_texts = np.concatenate([train_texts_final, pseudo_texts])
augmented_labels = np.concatenate([train_labels_final, pseudo_labels])

print(f"Pseudo-labels added: {len(pseudo_texts)} samples (confidence >= 0.95)")

# Create new dataset and dataloader for retraining
augmented_dataset = SpookyDataset(
    augmented_texts, augmented_labels, tokenizer, max_length, bt_texts=bt_texts, bt_prob=0.15
)
augmented_loader = DataLoader(
    augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

# Reduce learning rates for retraining (backbone 1e-6, head 2e-6)
retrain_optimizer = AdamW(
    [
        {
            "params": backbone_unfrozen_params,
            "lr": 1e-6,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        {"params": head_params, "lr": 2e-6, "weight_decay": 0.05, "betas": (0.9, 0.98)},
    ],
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

retrain_epochs = 10
retrain_patience = 5
best_retrain_score = float("inf")
epochs_no_improve_retrain = 0
scaler_retrain = GradScaler()

retrain_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    retrain_optimizer,
    max_lr=[1e-6, 2e-6],
    steps_per_epoch=1,
    epochs=retrain_epochs,
    pct_start=0.1,
    final_div_factor=1000,
)

for epoch in range(retrain_epochs):
    model.train()
    total_train_loss_pl = 0
    num_train_batches_pl = 0

    for batch_idx, batch in enumerate(augmented_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        retrain_optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler_retrain.scale(loss).backward()
        scaler_retrain.unscale_(retrain_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_retrain.step(retrain_optimizer)
        scaler_retrain.update()

        total_train_loss_pl += loss.item()
        num_train_batches_pl += 1

    retrain_scheduler.step()

    avg_train_loss_pl = total_train_loss_pl / num_train_batches_pl

    # Validate after each epoch
    model.eval()
    total_val_loss_pl = 0
    num_val_batches_pl = 0
    all_val_probs_pl = []
    all_val_labels_pl = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

            total_val_loss_pl += loss.item()
            num_val_batches_pl += 1
            all_val_probs_pl.append(probs.cpu().numpy())
            all_val_labels_pl.append(labels.cpu().numpy())

    avg_val_loss_pl = total_val_loss_pl / num_val_batches_pl

    val_probs_pl = np.concatenate(all_val_probs_pl, axis=0)
    val_true_pl = np.concatenate(all_val_labels_pl, axis=0)

    val_probs_clipped_pl = np.clip(val_probs_pl, 1e-15, 1 - 1e-15)
    val_probs_clipped_pl = val_probs_clipped_pl / val_probs_clipped_pl.sum(axis=1, keepdims=True)

    val_score_pl = log_loss(val_true_pl, val_probs_clipped_pl)

    current_lr = retrain_optimizer.param_groups[0]["lr"]
    print(
        f"Pseudo Epoch {epoch+1:2d}/{retrain_epochs} | Train Loss: {avg_train_loss_pl:.4f} | Val Loss: {avg_val_loss_pl:.4f} | Val LogLoss: {val_score_pl:.4f} | LR: {current_lr:.2e}"
    )

    if val_score_pl < best_retrain_score:
        best_retrain_score = val_score_pl
        epochs_no_improve_retrain = 0
        torch.save(model.state_dict(), "./working/best_model_pseudo.pt")
    else:
        epochs_no_improve_retrain += 1
        if epochs_no_improve_retrain >= retrain_patience:
            print(f"Early stopping triggered after pseudo epoch {epoch+1}")
            break

# Load best pseudo-trained model
model.load_state_dict(torch.load("./working/best_model_pseudo.pt"))
model.eval()

# ============================================================
# TEST INFERENCE (after pseudo-labeling)
# ============================================================
all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission_4cd8f67beff64d92a740688e38cd5073.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score (before pseudo): {final_val_score}")
print(f"Final Validation Score (after pseudo): {best_retrain_score}")