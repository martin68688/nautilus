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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

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

# ==================== MODEL CONFIGURATION ====================
print("Loading DeBERTa-v3-large model...")
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 256
batch_size = 8  # Smaller batch for large model
epochs = 5
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== CUSTOM MODEL CLASS ====================
class CustomDebertaModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super().__init__()
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
        )
        # Remove the original classifier head, we'll build our own
        hidden_size = self.deberta.config.hidden_size
        self.deberta.classifier = nn.Identity()

        # Statistical feature MLP
        self.stat_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )

        self.dropout_prob = dropout_prob

    def compute_stat_features(self, text):
        # Input is a single text string
        length = len(text)
        words = text.split()
        word_count = len(words)
        punct_count = sum(1 for c in text if c in '.,!?;:\'"-()[]{}')
        punct_density = punct_count / max(length, 1)
        cap_count = sum(1 for c in text if c.isupper())
        cap_ratio = cap_count / max(length, 1)
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those', 'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'as', 'just', 'also', 'very', 'too', 'really', 'quite', 'some', 'any', 'each', 'every', 'all', 'both', 'few', 'many', 'much', 'more', 'most', 'other', 'another'])
        stopword_count = sum(1 for w in words if w.lower() in stopwords)
        stopword_ratio = stopword_count / max(word_count, 1)

        return torch.tensor([length, word_count, punct_density, cap_ratio, stopword_ratio], dtype=torch.float)

    def forward(self, input_ids, attention_mask, texts=None, labels=None):
        # Get hidden states from DeBERTa
        outputs = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token embedding (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Compute statistical features if texts are provided
        if texts is not None:
            stat_features = torch.stack([self.compute_stat_features(t) for t in texts]).to(input_ids.device)
        else:
            stat_features = torch.zeros((input_ids.size(0), 5), device=input_ids.device)

        # Process statistical features through MLP
        stat_emb = self.stat_mlp(stat_features)

        # Concatenate and classify
        combined = torch.cat([pooled_output, stat_emb], dim=1)
        logits = self.classifier(combined)

        # Multi-Sample Dropout: run classifier head multiple times during training
        if self.training:
            all_logits = []
            for _ in range(4):
                # Re-apply dropout by passing through classifier again with different dropout mask
                logits_i = self.classifier(combined)
                all_logits.append(logits_i)
            logits = torch.stack(all_logits).mean(dim=0)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels)

        # Return in a format compatible with the rest of the code
        return type('Output', (), {'loss': loss, 'logits': logits})()


# ==================== DATASET CLASS ====================
# Synonym dictionary curated from author vocabulary
SYNONYM_DICT = {
    'dark': 'gloomy',
    'fear': 'dread',
    'strange': 'peculiar',
    'ancient': 'antique',
    'old': 'aged',
    'night': 'evening',
    'house': 'dwelling',
    'great': 'vast',
    'strange': 'odd',
    'terrible': 'dreadful',
    'sudden': 'abrupt',
    'cold': 'chilly',
    'darkness': 'gloom',
    'frightful': 'horrible',
    'silent': 'still',
    'door': 'portal',
    'water': 'stream',
    'saw': 'beheld',
    'thought': 'pondered',
    'place': 'spot',
    'power': 'might',
    'shadow': 'shade',
    'sound': 'noise',
    'heart': 'breast',
    'death': 'demise',
}

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, is_training=False):
        self.texts = texts
        self.labels = labels
        self.is_training = is_training

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Apply synonym replacement augmentation only during training
        if self.is_training and np.random.random() < 0.5:
            # Get list of keys present in the text
            keys_in_text = [k for k in SYNONYM_DICT.keys() if k in text.lower()]
            if keys_in_text:
                # Choose one random key and replace its first occurrence
                chosen_key = keys_in_text[np.random.randint(0, len(keys_in_text))]
                synonym = SYNONYM_DICT[chosen_key]
                # Case-insensitive replacement of first occurrence
                idx_lower = text.lower().find(chosen_key)
                if idx_lower != -1:
                    text = text[:idx_lower] + synonym + text[idx_lower + len(chosen_key):]

        encoding = tokenizer(
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
            return input_ids, attention_mask, label, self.texts[idx]  # Return original text unchanged for evaluation consistency
        return input_ids, attention_mask, text


# ==================== TRAINING FUNCTION ====================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, labels, texts) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, texts=texts, labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ==================== EVALUATION FUNCTION ====================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, texts in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, texts=texts)
                probs = torch.softmax(outputs.logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ==================== MAIN TRAINING LOOP ====================
print("Setting up stratified k-fold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []

# Store test predictions from each fold
all_fold_test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n===== Fold {fold+1} =====")

    fold_train_texts = [train_texts[i] for i in train_idx]
    fold_train_labels = train_labels[train_idx]
    fold_val_texts = [train_texts[i] for i in val_idx]
    fold_val_labels = train_labels[val_idx]

    # Create datasets and dataloaders (training data uses augmentation, validation does not)
    train_dataset = TextDataset(fold_train_texts, fold_train_labels, is_training=True)
    val_dataset = TextDataset(fold_val_texts, fold_val_labels, is_training=False)

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

    # Initialize custom model
    model = CustomDebertaModel(
        model_name,
        num_labels=num_authors,
        dropout_prob=0.1,
    )
    model.to(device)

    # Optimizer and scheduler with cosine annealing
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    # Cosine annealing scheduler with warmup
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device
        )

        val_preds, val_labels = evaluate(model, val_loader, device)

        # Clip probabilities as per competition requirements
        epsilon = 1e-15
        val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
        val_preds_clipped = val_preds_clipped / val_preds_clipped.sum(
            axis=1, keepdims=True
        )

        val_loss = log_loss(val_labels, val_preds_clipped)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Log Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model for this fold
            os.makedirs("./working", exist_ok=True)
            torch.save(model.state_dict(), f"./working/best_model_fold_{fold}.pt")

    val_scores.append(best_val_loss)
    print(f"Fold {fold+1} best validation log loss: {best_val_loss:.4f}")

    # Generate test predictions for this fold's best model
    print(f"Generating test predictions for fold {fold+1}...")
    fold_model = CustomDebertaModel(
        model_name,
        num_labels=num_authors,
        dropout_prob=0.1,
    )
    fold_model.load_state_dict(torch.load(f"./working/best_model_fold_{fold}.pt", map_location=device))
    fold_model.to(device)
    fold_model.eval()

    test_dataset = TextDataset(test_texts, is_training=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    fold_test_preds = []
    with torch.no_grad():
        for input_ids, attention_mask, texts in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask, texts=texts)
                probs = torch.softmax(outputs.logits, dim=1)

            fold_test_preds.append(probs.cpu().numpy())

    fold_test_preds = np.concatenate(fold_test_preds, axis=0)
    all_fold_test_preds.append(fold_test_preds)

    # Clean up GPU memory
    del fold_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nAverage validation log loss across folds: {np.mean(val_scores):.4f}")

# ==================== ENSEMBLE TEST PREDICTIONS ====================
print("\n===== Ensembling test predictions from all folds =====")
test_preds = np.mean(all_fold_test_preds, axis=0)

# Clip and normalize probabilities
epsilon = 1e-15
test_preds_clipped = np.clip(test_preds, epsilon, 1 - epsilon)
test_preds_clipped = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

# ==================== SUBMISSION ====================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_clipped[:, 0],
        "HPL": test_preds_clipped[:, 1],
        "MWS": test_preds_clipped[:, 2],
    }
)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

final_val_score = np.mean(val_scores)
print(f"Final Validation Score: {final_val_score}")