import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
import re
import os
import gc
import warnings
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# 2. TEXT CLEANING
# ============================================================
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"\s+", " ", text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("—", "--").replace("–", "-")
        return text.strip()
    return str(text) if pd.notna(text) else ""


print("Cleaning text...")
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# ============================================================
# 3. STYLOMETRIC FEATURE ENGINEERING
# ============================================================
print("Engineering stylometric features...")


def extract_stylometric_features(text):
    features = {}
    features["char_count"] = len(text)
    features["word_count"] = len(text.split())
    features["sentence_count"] = len(re.findall(r"[.!?]+", text)) + 1
    features["avg_word_length"] = (
        np.mean([len(w) for w in text.split()]) if text.split() else 0
    )
    words = text.lower().split()
    features["unique_word_ratio"] = len(set(words)) / max(len(words), 1)
    features["comma_count"] = text.count(",")
    features["semicolon_count"] = text.count(";")
    features["colon_count"] = text.count(":")
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["dash_count"] = text.count("-") + text.count("--")
    features["quote_count"] = (
        text.count('"') + text.count("\u201c") + text.count("\u201d")
    )
    features["period_count"] = text.count(".")
    total_punct = sum(
        [
            features["comma_count"],
            features["semicolon_count"],
            features["colon_count"],
            features["exclamation_count"],
            features["question_count"],
            features["dash_count"],
            features["quote_count"],
            features["period_count"],
        ]
    )
    features["punctuation_density"] = total_punct / max(features["char_count"], 1)
    avg_sentence_len_words = features["word_count"] / max(features["sentence_count"], 1)
    avg_sentence_len_chars = features["char_count"] / max(features["sentence_count"], 1)
    features["avg_sentence_word_len"] = avg_sentence_len_words
    features["avg_sentence_char_len"] = avg_sentence_len_chars
    features["capitalized_word_count"] = sum(
        1 for w in text.split() if w and w[0].isupper()
    )
    features["all_caps_word_count"] = sum(
        1 for w in text.split() if w.isupper() and len(w) > 1
    )
    features["capital_ratio"] = features["capitalized_word_count"] / max(
        features["word_count"], 1
    )
    features["ellipsis_count"] = text.count("...") + text.count(". . .")
    features["ampersand_count"] = text.count("&")
    features["parentheses_count"] = text.count("(") + text.count(")")
    features["double_punct_count"] = len(re.findall(r"[!?]{2,}", text))

    function_words = [
        "the",
        "and",
        "a",
        "to",
        "of",
        "in",
        "was",
        "that",
        "had",
        "his",
        "with",
        "not",
        "but",
        "all",
        "from",
        "by",
        "or",
        "as",
        "at",
        "for",
        "an",
        "were",
        "which",
        "have",
        "this",
        "been",
        "has",
        "are",
        "would",
        "so",
        "my",
        "no",
        "upon",
        "its",
        "more",
        "could",
        "she",
        "before",
        "than",
        "what",
        "other",
        "into",
        "much",
        "very",
        "may",
        "should",
        "most",
        "every",
        "through",
        "these",
    ]
    word_freq = pd.Series(text.lower().split()).value_counts()
    for word in function_words:
        features[f"fw_{word}"] = word_freq.get(word, 0) / max(features["word_count"], 1)

    archaic_words = [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "dost",
        "whence",
        "thence",
        "hither",
        "thither",
        "wherefore",
        "henceforth",
        "heretofore",
        "thereunto",
        "wherewith",
    ]
    features["archaic_word_count"] = sum(
        1 for w in text.lower().split() if w in archaic_words
    )
    features["archaic_word_ratio"] = features["archaic_word_count"] / max(
        features["word_count"], 1
    )

    horror_words_poe = [
        "dark",
        "night",
        "death",
        "dead",
        "grave",
        "shadow",
        "terror",
        "fear",
        "horror",
        "dread",
        "gloom",
        "weep",
        "sorrow",
        "mourn",
        "despair",
        "corpse",
        "coffin",
        "decay",
        "rot",
        "sepulchre",
        "vault",
        "tomb",
    ]
    horror_words_lovecraft = [
        "eldritch",
        "cosmic",
        "ancient",
        "unspeakable",
        "nameless",
        "cyclopean",
        "cavernous",
        "abyss",
        "primordial",
        "creature",
        "monstrous",
        "loathsome",
        "crawling",
        "slimy",
        "blasphemous",
        "miasmatic",
        "gibbous",
        "non-euclidean",
        "squamous",
        "rugose",
    ]
    horror_words_shelley = [
        "monster",
        "creature",
        "science",
        "nature",
        "power",
        "knowledge",
        "life",
        "death",
        "spirit",
        "soul",
        "mystery",
        "secret",
        "experiment",
        "creation",
    ]
    features["poe_horror_score"] = sum(
        1 for w in text.lower().split() if w in horror_words_poe
    )
    features["lovecraft_horror_score"] = sum(
        1 for w in text.lower().split() if w in horror_words_lovecraft
    )
    features["shelley_horror_score"] = sum(
        1 for w in text.lower().split() if w in horror_words_shelley
    )

    first_person_pronouns = [
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "us",
        "our",
        "ours",
    ]
    third_person_pronouns = [
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
    first_person_count = sum(
        1 for w in text.lower().split() if w in first_person_pronouns
    )
    third_person_count = sum(
        1 for w in text.lower().split() if w in third_person_pronouns
    )
    features["first_person_ratio"] = first_person_count / max(
        first_person_count + third_person_count, 1
    )

    return features


print("Extracting features for training data...")
train_features = train_df["clean_text"].apply(extract_stylometric_features)
train_feature_df = pd.DataFrame(train_features.tolist())

print("Extracting features for test data...")
test_features = test_df["clean_text"].apply(extract_stylometric_features)
test_feature_df = pd.DataFrame(test_features.tolist())

print(
    f"Stylometric features shape - Train: {train_feature_df.shape}, Test: {test_feature_df.shape}"
)

# ============================================================
# 4. N-GRAM FEATURES
# ============================================================
print("Creating n-gram features...")

char_vectorizer = CountVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=500, min_df=3, max_df=0.9
)
print("Fitting character n-grams...")
train_char_feats = char_vectorizer.fit_transform(train_df["clean_text"])
test_char_feats = char_vectorizer.transform(test_df["clean_text"])
print(
    f"Character n-gram features - Train: {train_char_feats.shape}, Test: {test_char_feats.shape}"
)

word_vectorizer = CountVectorizer(
    analyzer="word", ngram_range=(1, 3), max_features=1000, min_df=5, max_df=0.8
)
print("Fitting word n-grams...")
train_word_feats = word_vectorizer.fit_transform(train_df["clean_text"])
test_word_feats = word_vectorizer.transform(test_df["clean_text"])
print(
    f"Word n-gram features - Train: {train_word_feats.shape}, Test: {test_word_feats.shape}"
)

tfidf_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=2000,
    min_df=3,
    max_df=0.85,
    sublinear_tf=True,
)
print("Fitting TF-IDF features...")
train_tfidf_feats = tfidf_vectorizer.fit_transform(train_df["clean_text"])
test_tfidf_feats = tfidf_vectorizer.transform(test_df["clean_text"])
print(
    f"TF-IDF features - Train: {train_tfidf_feats.shape}, Test: {test_tfidf_feats.shape}"
)

# ============================================================
# 5. COMBINE ALL FEATURES
# ============================================================
print("Combining all features...")

train_stylo_sparse = csr_matrix(train_feature_df.fillna(0).values)
test_stylo_sparse = csr_matrix(test_feature_df.fillna(0).values)

train_features_combined = hstack(
    [train_stylo_sparse, train_char_feats, train_word_feats, train_tfidf_feats]
).tocsr()
test_features_combined = hstack(
    [test_stylo_sparse, test_char_feats, test_word_feats, test_tfidf_feats]
).tocsr()

print(
    f"Combined features shape - Train: {train_features_combined.shape}, Test: {test_features_combined.shape}"
)

# ============================================================
# 6. TARGET ENCODING AND SPLIT
# ============================================================
print("Preparing target variable...")
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)
print(f"Classes: {label_encoder.classes_}, Encoded: {list(range(num_classes))}")

print("Creating train/validation split...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(
    train_features_combined, train_df["author_encoded"]
):
    train_idx = train_idx
    val_idx = val_idx
    break

X_train = train_features_combined[train_idx]
y_train = train_df["author_encoded"].iloc[train_idx].values
X_val = train_features_combined[val_idx]
y_val = train_df["author_encoded"].iloc[val_idx].values

print(
    f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {test_features_combined.shape[0]}"
)

# ============================================================
# 7. SAVE INTERMEDIATE DATA
# ============================================================
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/X_train.npy", X_train.toarray())
np.save(f"{output_dir}/y_train.npy", y_train)
np.save(f"{output_dir}/X_val.npy", X_val.toarray())
np.save(f"{output_dir}/y_val.npy", y_val)
np.save(f"{output_dir}/X_test.npy", test_features_combined.toarray())

joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")
joblib.dump(char_vectorizer, f"{output_dir}/char_vectorizer.pkl")
joblib.dump(word_vectorizer, f"{output_dir}/word_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, f"{output_dir}/tfidf_vectorizer.pkl")

# Save indices for later use
np.save(f"{output_dir}/train_idx.npy", train_idx)
np.save(f"{output_dir}/val_idx.npy", val_idx)


# ============================================================
# 8. DATASET CLASS
# ============================================================
class AuthorshipDataset(Dataset):
    def __init__(
        self, texts, stylo_features, labels=None, tokenizer=None, max_length=256
    ):
        self.texts = texts
        self.stylo_features = stylo_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        stylo = torch.FloatTensor(self.stylo_features[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.LongTensor(encoding["input_ids"]),
            "attention_mask": torch.LongTensor(encoding["attention_mask"]),
            "stylo_features": stylo,
        }
        if self.labels is not None:
            item["labels"] = torch.LongTensor([self.labels[idx]])[0]
        return item


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    stylo_features = torch.stack([item["stylo_features"] for item in batch])
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "stylo_features": stylo_features,
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


# ============================================================
# 9. MODEL DEFINITION
# ============================================================
class HybridAuthorshipModel(nn.Module):
    def __init__(self, num_classes=3, num_stylometric_features=100, dropout_rate=0.2):
        super(HybridAuthorshipModel, self).__init__()
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        self.deberta_hidden_size = self.deberta.config.hidden_size

        self.stylo_projection = nn.Sequential(
            nn.Linear(num_stylometric_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        combined_size = self.deberta_hidden_size + 256
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [self.stylo_projection, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, input_ids, attention_mask, stylo_features):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        stylo_emb = self.stylo_projection(stylo_features)
        combined = torch.cat([cls_embedding, stylo_emb], dim=1)
        logits = self.classifier(combined)
        return logits


# ============================================================
# 10. PREPARE DATA LOADERS
# ============================================================
print("Preparing data for training...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = AuthorshipDataset(
    texts=train_df["clean_text"].iloc[train_idx].values,
    stylo_features=X_train.toarray(),
    labels=y_train,
    tokenizer=tokenizer,
    max_length=256,
)

val_dataset = AuthorshipDataset(
    texts=train_df["clean_text"].iloc[val_idx].values,
    stylo_features=X_val.toarray(),
    labels=y_val,
    tokenizer=tokenizer,
    max_length=256,
)

test_dataset = AuthorshipDataset(
    texts=test_df["clean_text"].values,
    stylo_features=test_features_combined.toarray(),
    labels=None,
    tokenizer=tokenizer,
    max_length=256,
)

print(
    f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)

# ============================================================
# 11. MODEL, OPTIMIZER, LOSS, SCHEDULER SETUP
# ============================================================
num_stylometric_features = X_train.shape[1]
model = HybridAuthorshipModel(
    num_classes=3, num_stylometric_features=num_stylometric_features, dropout_rate=0.3
)
model.to(device)

criterion = nn.CrossEntropyLoss()

backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "deberta" in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW(
    [{"params": backbone_params, "lr": 2e-5}, {"params": head_params, "lr": 2e-4}],
    weight_decay=0.01,
)

total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS

from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
)

scaler = GradScaler()

# ============================================================
# 12. TRAINING LOOP
# ============================================================
print("Starting training...")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Total optimization steps: {total_steps}")

best_val_loss = float("inf")
best_log_loss = float("inf")
best_model_state = None
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_features = batch["stylo_features"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        train_batches += 1

    avg_train_loss = train_loss / train_batches

    # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_features = batch["stylo_features"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, stylo_features)
                loss = criterion(logits, labels)

            val_loss += loss.item()
            val_batches += 1
            probabilities = torch.softmax(logits, dim=1)
            all_val_preds.append(probabilities.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_labels = np.concatenate(all_val_labels, axis=0)

    # Calculate multiclass log loss
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(
        axis=1, keepdims=True
    )
    N = len(val_labels)
    M = 3
    log_loss_val = 0.0
    for i in range(N):
        for j in range(M):
            y_ij = 1 if val_labels[i] == j else 0
            log_loss_val += y_ij * np.log(val_preds_normalized[i, j])
    log_loss_val = -log_loss_val / N

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Log Loss: {log_loss_val:.4f}"
    )

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_log_loss = log_loss_val
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 13. LOAD BEST MODEL AND FINAL VALIDATION SCORE
# ============================================================
print("Loading best model...")
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

all_val_preds = []
all_val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_features = batch["stylo_features"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
        probabilities = torch.softmax(logits, dim=1)
        all_val_preds.append(probabilities.cpu().numpy())
        all_val_labels.append(labels.cpu().numpy())

val_preds = np.concatenate(all_val_preds, axis=0)
val_labels = np.concatenate(all_val_labels, axis=0)

epsilon = 1e-15
val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
val_preds_normalized = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)
N = len(val_labels)
M = 3
final_log_loss = 0.0
for i in range(N):
    for j in range(M):
        y_ij = 1 if val_labels[i] == j else 0
        final_log_loss += y_ij * np.log(val_preds_normalized[i, j])
final_log_loss = -final_log_loss / N

print(f"Final Validation Score: {final_log_loss}")

# ============================================================
# 14. TEST INFERENCE
# ============================================================
print("Running test inference...")
model.eval()
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stylo_features = batch["stylo_features"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask, stylo_features)
        probabilities = torch.softmax(logits, dim=1)
        all_test_preds.append(probabilities.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0)
print(f"Test predictions shape: {test_preds.shape}")

# ============================================================
# 15. CREATE SUBMISSION FILE
# ============================================================
print("Creating submission file...")
test_ids = test_df["id"].values

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
print(f"First 5 rows:\n{submission_df.head()}")
