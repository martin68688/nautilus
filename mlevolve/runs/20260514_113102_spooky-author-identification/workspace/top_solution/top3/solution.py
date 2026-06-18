import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
import math

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

os.makedirs("./working", exist_ok=True)

# ============================================================
# MODEL DESIGN - DeBERTa-v3-large with Custom Head (from Step 2)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    """DeBERTa-v3-large with intermediate projection and multi-sample dropout"""

    def __init__(self, num_authors=3, dropout_rate=0.2, hidden_dim=256, n_dropouts=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            output_hidden_states=True,
            output_attentions=False,
        )

        # Freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 8 layers for fine-tuning
        for layer in self.backbone.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.hidden_size = self.backbone.config.hidden_size

        # Intermediate projection layer (now 3x hidden_size from concatenation)
        self.intermediate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Multi-sample dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_dropouts)]
        )
        self.classifier = nn.Linear(hidden_dim, num_authors)

        self._init_weights()

    def _init_weights(self):
        for module in [self.intermediate, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, manual_features=None, return_aux=False, return_dropout_samples=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # Multi-Head Attention + Mean-Max Concatenation pooling
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Max pooling
        mask_expanded_inf = (1.0 - mask_expanded) * -1e9
        max_pooled = torch.max(last_hidden + mask_expanded_inf, dim=1)[0]

        # Multi-Head Attention pooling - use cls token as query
        if not hasattr(self, 'attention_pool'):
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.hidden_size, num_heads=8, batch_first=True
            ).to(last_hidden.device)

        cls_token = last_hidden[:, 0:1, :]  # (batch, 1, hidden_size)
        attn_output, _ = self.attention_pool(cls_token, last_hidden, last_hidden, key_padding_mask=(1 - attention_mask).bool())
        attn_pooled = attn_output.squeeze(1)  # (batch, hidden_size)

        # Concatenate mean, max, and attention pooled features
        pooled = torch.cat([mean_pooled, max_pooled, attn_pooled], dim=-1)  # (batch, hidden_size*3)

        features = self.intermediate(pooled)

        if return_aux and manual_features is not None:
            # Create auxiliary logits from handcrafted features
            if not hasattr(self, 'aux_projection'):
                self.aux_projection = nn.Sequential(
                    nn.Linear(manual_features.size(-1), self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size, 3)
                ).to(features.device)
            aux_logits = self.aux_projection(manual_features)

        if return_dropout_samples:
            logits_list = []
            for dropout in self.dropouts:
                dropped = dropout(features)
                logits = self.classifier(dropped)
                logits_list.append(logits)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            logits = self.classifier(features)

        if return_aux and manual_features is not None:
            return logits, aux_logits
        else:
            return logits


model = SpookyAuthorClassifier(
    num_authors=3, dropout_rate=0.2, hidden_dim=256, n_dropouts=4
)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Differential learning rates
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
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 5e-5, "weight_decay": 0.01},
    ],
    betas=(0.9, 0.999),
    eps=1e-8,
)

print(f"Backbone unfrozen params: {sum(p.numel() for p in backbone_params):,}")
print(f"Head params: {sum(p.numel() for p in head_params):,}")
print(
    f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# DATASET AND DATALOADER
# ============================================================
def extract_manual_features(text):
    """Extract handcrafted features from text: lexical diversity, syntactic, readability, stylistic markers."""
    features = []

    # Basic text stats
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chars = len(text)
    num_words = len(words)
    num_sentences = len(sentences)
    num_chars = chars

    # Lexical diversity features (5)
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / max(num_words, 1)
    features.append(lexical_diversity)

    # Type-token ratio for first 50 words
    first_50 = [w.lower() for w in words[:50]]
    ttr_50 = len(set(first_50)) / max(len(first_50), 1)
    features.append(ttr_50)

    # Hapax legomena ratio (words appearing once)
    word_freq = {}
    for w in words:
        w_lower = w.lower()
        word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
    hapax_count = sum(1 for v in word_freq.values() if v == 1)
    hapax_ratio = hapax_count / max(num_words, 1)
    features.append(hapax_ratio)

    # Syntactic features (8)
    avg_word_length = num_chars / max(num_words, 1)
    features.append(avg_word_length)

    avg_sentence_length = num_words / max(num_sentences, 1)
    features.append(avg_sentence_length)

    # Punctuation density
    punct_count = sum(1 for c in text if c in '.,;:!?\'"()-')
    punct_density = punct_count / max(num_chars, 1)
    features.append(punct_density)

    # Comma frequency
    comma_count = text.count(',')
    comma_density = comma_count / max(num_words, 1)
    features.append(comma_density)

    # Exclamation/question marks
    excl_count = text.count('!')
    ques_count = text.count('?')
    excl_density = excl_count / max(num_sentences, 1)
    ques_density = ques_count / max(num_sentences, 1)
    features.extend([excl_density, ques_density])

    # Capitalization ratio
    cap_words = sum(1 for w in words if w[0].isupper() if len(w) > 0)
    cap_ratio = cap_words / max(num_words, 1)
    features.append(cap_ratio)

    # Readability features (6)
    # Flesch-Kincaid readability proxy
    syllables = 0
    for w in words:
        w_lower = w.lower()
        if len(w_lower) <= 3:
            syllables += 1
        else:
            # Count vowel groups
            vowel_count = sum(1 for i, c in enumerate(w_lower) if c in 'aeiou' and (i == 0 or w_lower[i-1] not in 'aeiou'))
            syllables += max(1, vowel_count)

    avg_syllables_per_word = syllables / max(num_words, 1)
    features.append(avg_syllables_per_word)

    # Flesch Reading Ease proxy (higher = easier)
    flesch = 206.835 - 1.015 * (num_words / max(num_sentences, 1)) - 84.6 * (syllables / max(num_words, 1))
    features.append(flesch / 100.0)  # normalize

    # Automated Readability Index proxy
    ari = 4.71 * (num_chars / max(num_words, 1)) + 0.5 * (num_words / max(num_sentences, 1)) - 21.43
    features.append(ari / 20.0)  # normalize

    # Coleman-Liau Index proxy
    l = (num_chars / max(num_words, 1)) * 100
    s = (num_sentences / max(num_words, 1)) * 100
    cli = 0.0588 * l - 0.296 * s - 15.8
    features.append(cli / 20.0)  # normalize

    # Stylistic markers (15)
    # Function word ratios
    function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'can', 'could', 'should', 'may', 'might', 'shall', 'that',
                     'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its',
                     'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'also', 'very',
                     'just', 'only', 'still', 'even', 'too', 'much', 'many', 'some',
                     'any', 'each', 'every', 'all', 'both', 'neither', 'either'}

    func_word_count = sum(1 for w in words if w.lower() in function_words)
    func_word_ratio = func_word_count / max(num_words, 1)
    features.append(func_word_ratio)

    # First person pronoun ratio
    first_person = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
    first_person_count = sum(1 for w in words if w.lower() in first_person)
    first_person_ratio = first_person_count / max(num_words, 1)
    features.append(first_person_ratio)

    # Second person pronoun ratio
    second_person = {'you', 'your', 'yours', 'yourself'}
    second_person_count = sum(1 for w in words if w.lower() in second_person)
    second_person_ratio = second_person_count / max(num_words, 1)
    features.append(second_person_ratio)

    # Third person pronoun ratio
    third_person = {'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs'}
    third_person_count = sum(1 for w in words if w.lower() in third_person)
    third_person_ratio = third_person_count / max(num_words, 1)
    features.append(third_person_ratio)

    # Past tense verbs (simple heuristic)
    past_tense_count = sum(1 for w in words if w.lower().endswith('ed'))
    past_tense_ratio = past_tense_count / max(num_words, 1)
    features.append(past_tense_ratio)

    # Present tense verbs (simple heuristic)
    present_tense_count = sum(1 for w in words if w.lower().endswith('s') and not w.lower().endswith('ss'))
    present_tense_ratio = present_tense_count / max(num_words, 1)
    features.append(present_tense_ratio)

    # Adverb frequency (words ending in 'ly')
    adverb_count = sum(1 for w in words if w.lower().endswith('ly'))
    adverb_ratio = adverb_count / max(num_words, 1)
    features.append(adverb_ratio)

    # Contraction frequency
    contraction_count = sum(1 for w in words if "'" in w)
    contraction_ratio = contraction_count / max(num_words, 1)
    features.append(contraction_ratio)

    # Quotation density
    quote_count = text.count('"') // 2  # count pairs
    quote_density = quote_count / max(num_sentences, 1)
    features.append(quote_density)

    # Dash/hyphen frequency
    dash_count = text.count('--') + text.count('-')
    dash_density = dash_count / max(num_words, 1)
    features.append(dash_density)

    # Ellipsis count
    ellipsis_count = text.count('...')
    ellipsis_density = ellipsis_count / max(num_sentences, 1)
    features.append(ellipsis_density)

    # Semicolon frequency
    semicolon_count = text.count(';')
    semicolon_density = semicolon_count / max(num_words, 1)
    features.append(semicolon_density)

    # Colon frequency
    colon_count = text.count(':')
    colon_density = colon_count / max(num_words, 1)
    features.append(colon_density)

    # Part-of-speech style features (simple heuristics)
    # Words ending in 'ing' (gerunds/participles)
    ing_count = sum(1 for w in words if w.lower().endswith('ing'))
    ing_ratio = ing_count / max(num_words, 1)
    features.append(ing_ratio)

    # Words ending in 'tion' or 'sion' (nominalizations)
    tion_count = sum(1 for w in words if w.lower().endswith('tion') or w.lower().endswith('sion'))
    tion_ratio = tion_count / max(num_words, 1)
    features.append(tion_ratio)

    # Sentiment proxy: positive vs negative word ratio (simplified)
    positive_words = {'good', 'great', 'beautiful', 'wonderful', 'happy', 'joy', 'love',
                     'excellent', 'perfect', 'amazing', 'bright', 'brilliant', 'kind',
                     'pleasant', 'delightful', 'elegant', 'peaceful', 'calm'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'evil', 'fear',
                     'dark', 'cold', 'pain', 'ugly', 'cruel', 'dreadful', 'sad',
                     'angry', 'bitter', 'gloomy', 'hopeless', 'terrifying'}
    pos_count = sum(1 for w in words if w.lower() in positive_words)
    neg_count = sum(1 for w in words if w.lower() in negative_words)
    sentiment_ratio = (pos_count - neg_count) / max(num_words, 1)
    features.append(sentiment_ratio)

    # Emotional intensity proxy
    emotion_words = positive_words | negative_words
    emotion_ratio = sum(1 for w in words if w.lower() in emotion_words) / max(num_words, 1)
    features.append(emotion_ratio)

    # To ensure exactly 40 features, pad or truncate
    while len(features) < 40:
        features.append(0.0)
    features = features[:40]

    return np.array(features, dtype=np.float32)


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-extract manual features
        self.manual_features = []
        for text in texts:
            self.manual_features.append(extract_manual_features(str(text)))
        self.manual_features = np.array(self.manual_features, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            "manual_features": torch.tensor(self.manual_features[idx], dtype=torch.float32),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Prepare data arrays
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

train_indices = train_set.index.tolist()
val_indices = val_set.index.tolist()

train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

batch_size = 16
max_length = 512

train_dataset = SpookyDataset(
    train_texts_final, train_labels_final, tokenizer, max_length
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
os.makedirs("./submission", exist_ok=True)

# OneCycleLR scheduler (per-epoch)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[2e-5, 5e-5],  # max_lr for backbone and head
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.1,
    final_div_factor=1000,
    div_factor=25,
)

# Auxiliary loss weight
aux_loss_lambda = 0.3

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        manual_features = batch["manual_features"].to(device)

        optimizer.zero_grad()
        with autocast():
            logits, aux_logits = model(input_ids, attention_mask, manual_features=manual_features, return_aux=True)
            main_loss = criterion(logits, labels)
            aux_loss = criterion(aux_logits, labels)
            loss = main_loss + aux_loss_lambda * aux_loss

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        scheduler.step()

        total_train_loss += loss.item()
        num_train_batches += 1

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
            manual_features = batch["manual_features"].to(device)

            with autocast():
                logits, aux_logits = model(input_ids, attention_mask, manual_features=manual_features, return_aux=True)
                main_loss = criterion(logits, labels)
                aux_loss = criterion(aux_logits, labels)
                loss = main_loss + aux_loss_lambda * aux_loss
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
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# ============================================================
# LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        manual_features = batch["manual_features"].to(device)
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
# TEST INFERENCE
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

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_val_score}")