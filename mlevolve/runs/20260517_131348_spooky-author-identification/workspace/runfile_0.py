import os
os.sched_setaffinity(0, {29, 30, 31})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re
import json

# -------------------- Load data --------------------
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# -------------------- Feature engineering --------------------
def extract_stylometric_features(text_series):
    features = []
    for text in text_series:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0, 0, 0, 0, 0, 0, 0, 0])
            continue
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        char_len = len(text)
        word_len = len(words)
        sent_len = len(sentences)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        comma_count = text.count(",")
        exclamation_count = text.count("!")
        question_count = text.count("?")
        punct_count = sum(1 for c in text if c in ".,;:!?\"'()[]-")
        punct_density = punct_count / char_len if char_len > 0 else 0
        features.append(
            [
                char_len,
                word_len,
                sent_len,
                avg_word_len,
                comma_count,
                exclamation_count + question_count,
                punct_count,
                punct_density,
            ]
        )
    return np.array(features)

archaic_words = set(
    [
        "thou",
        "thee",
        "thy",
        "thine",
        "hath",
        "doth",
        "art",
        "wilt",
        "shalt",
        "whence",
        "thence",
        "hither",
        "thither",
        "ere",
        "betwixt",
        "aforesaid",
        "perchance",
        "methinks",
        "anon",
        "wherefore",
        "hark",
        "alas",
        "forsooth",
        "prithee",
        "yonder",
        "fain",
        "deign",
        "durst",
        "twas",
        "tis",
    ]
)

def count_archaic(text):
    if not isinstance(text, str):
        return 0
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return sum(1 for w in words if w in archaic_words)

def uppercase_word_ratio(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    words = re.findall(r"\b[A-Za-z]+\b", text)
    if len(words) == 0:
        return 0.0
    upper_count = sum(1 for w in words if w[0].isupper())
    return upper_count / len(words)

def unique_word_ratio(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

train_stylometric = extract_stylometric_features(train_df["text"])
train_archaic = np.array([count_archaic(t) for t in train_df["text"]]).reshape(-1, 1)
train_upper = np.array([uppercase_word_ratio(t) for t in train_df["text"]]).reshape(
    -1, 1
)
train_unique = np.array([unique_word_ratio(t) for t in train_df["text"]]).reshape(-1, 1)
train_handcrafted = np.hstack(
    [train_stylometric, train_archaic, train_upper, train_unique]
)

test_stylometric = extract_stylometric_features(test_df["text"])
test_archaic = np.array([count_archaic(t) for t in test_df["text"]]).reshape(-1, 1)
test_upper = np.array([uppercase_word_ratio(t) for t in test_df["text"]]).reshape(-1, 1)
test_unique = np.array([unique_word_ratio(t) for t in test_df["text"]]).reshape(-1, 1)
test_handcrafted = np.hstack([test_stylometric, test_archaic, test_upper, test_unique])

# -------------------- Tokenization for DeBERTa --------------------
model_id = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_texts(texts, max_length=512):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    return encodings["input_ids"], encodings["attention_mask"]

train_input_ids, train_attention_masks = tokenize_texts(train_df["text"])
test_input_ids, test_attention_masks = tokenize_texts(test_df["text"])

# -------------------- Stratified split --------------------
label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = train_df["author"].map(label_map).values

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_labels
)

assert len(set(train_idx) & set(val_idx)) == 0, "Leakage detected!"

X_train_ids = train_input_ids[train_idx]
X_train_mask = train_attention_masks[train_idx]
X_train_hand = train_handcrafted[train_idx]
y_train = train_labels[train_idx]

X_val_ids = train_input_ids[val_idx]
X_val_mask = train_attention_masks[val_idx]
X_val_hand = train_handcrafted[val_idx]
y_val = train_labels[val_idx]

X_test_ids = test_input_ids
X_test_mask = test_attention_masks
X_test_hand = test_handcrafted

# -------------------- Save processed data --------------------
os.makedirs("./working", exist_ok=True)

np.save("./working/X_train_ids.npy", X_train_ids)
np.save("./working/X_train_mask.npy", X_train_mask)
np.save("./working/X_train_hand.npy", X_train_hand)
np.save("./working/y_train.npy", y_train)

np.save("./working/X_val_ids.npy", X_val_ids)
np.save("./working/X_val_mask.npy", X_val_mask)
np.save("./working/X_val_hand.npy", X_val_hand)
np.save("./working/y_val.npy", y_val)

np.save("./working/X_test_ids.npy", X_test_ids)
np.save("./working/X_test_mask.npy", X_test_mask)
np.save("./working/X_test_hand.npy", X_test_hand)

with open("./working/label_map.json", "w") as f:
    json.dump(label_map, f)

np.save("./working/test_ids.npy", test_df["id"].values)

train_processed = train_df.iloc[train_idx].copy()
val_processed = train_df.iloc[val_idx].copy()
test_processed = test_df.copy()

train_processed.to_csv("./working/train_processed.csv", index=False)
val_processed.to_csv("./working/val_processed.csv", index=False)
test_processed.to_csv("./working/test_processed.csv", index=False)

print(
    f"Data processing complete. Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_df)}"
)

# -------------------- Model Architecture --------------------
num_labels = 3

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

for name, param in model.deberta.named_parameters():
    param.requires_grad = False

for i in range(22, 24):
    for param in model.deberta.encoder.layer[i].parameters():
        param.requires_grad = True

for name, param in model.deberta.named_parameters():
    if "LayerNorm" in name:
        param.requires_grad = True

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(
    [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": 5e-5,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "classifier" not in n and p.requires_grad
            ],
            "lr": 2e-5,
        },
    ],
    weight_decay=0.01,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model: {model_id} with {num_labels} classes")
print(f"Device: {device}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------- Training Setup --------------------
batch_size = 8
num_workers = 2 if device.type == "cuda" else 0

train_dataset = TensorDataset(
    torch.tensor(X_train_ids, dtype=torch.long),
    torch.tensor(X_train_mask, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.long),
)

val_dataset = TensorDataset(
    torch.tensor(X_val_ids, dtype=torch.long),
    torch.tensor(X_val_mask, dtype=torch.long),
    torch.tensor(y_val, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

scaler = GradScaler(enabled=(device.type == "cuda"))

num_epochs = 10
gradient_accumulation_steps = 4
patience = 3
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

# -------------------- Training Loop --------------------
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    num_train_batches = 0

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_train_loss += loss.item() * gradient_accumulation_steps
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    # Validation phase
    model.eval()
    val_preds = []
    val_labels_list = []
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits

            total_val_loss += loss.item()
            num_val_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            val_preds.append(probs)
            val_labels_list.append(labels.cpu().numpy())

    avg_val_loss = total_val_loss / num_val_batches

    val_preds = np.concatenate(val_preds, axis=0)
    val_labels_concat = np.concatenate(val_labels_list, axis=0)

    epsilon = 1e-15
    val_preds_clamped = np.clip(val_preds, epsilon, 1 - epsilon)
    val_preds_normalized = val_preds_clamped / val_preds_clamped.sum(
        axis=1, keepdims=True
    )

    val_logloss = log_loss(val_labels_concat, val_preds_normalized)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f}"
    )

    if avg_val_loss + 0.001 < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_logloss = val_logloss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"  -> New best model! Val LogLoss: {val_logloss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print("Loading best model for final evaluation and test inference...")
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

# -------------------- Final Validation Evaluation --------------------
val_preds = []
val_labels_list = []

with torch.no_grad():
    for input_ids, attention_mask, labels in val_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        val_preds.append(probs)
        val_labels_list.append(labels.numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_labels_concat = np.concatenate(val_labels_list, axis=0)

epsilon = 1e-15
val_preds_clamped = np.clip(val_preds, epsilon, 1 - epsilon)
val_preds_normalized = val_preds_clamped / val_preds_clamped.sum(axis=1, keepdims=True)

final_val_logloss = log_loss(val_labels_concat, val_preds_normalized)
print(f"Final Validation LogLoss: {final_val_logloss:.6f}")

# -------------------- Test Inference --------------------
print("Performing test inference...")
test_dataset = TensorDataset(
    torch.tensor(X_test_ids, dtype=torch.long),
    torch.tensor(X_test_mask, dtype=torch.long),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_preds = []
with torch.no_grad():
    for input_ids, attention_mask in test_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        test_preds.append(probs)

test_preds = np.concatenate(test_preds, axis=0)

test_preds_clamped = np.clip(test_preds, epsilon, 1 - epsilon)
test_preds_normalized = test_preds_clamped / test_preds_clamped.sum(
    axis=1, keepdims=True
)

# -------------------- Create Submission --------------------
os.makedirs("./submission", exist_ok=True)

author_names = ["EAP", "HPL", "MWS"]
submission_df = pd.DataFrame(
    {
        "id": np.load("./working/test_ids.npy"),
        author_names[0]: test_preds_normalized[:, 0],
        author_names[1]: test_preds_normalized[:, 1],
        author_names[2]: test_preds_normalized[:, 2],
    }
)

submission_df.to_csv("./submission/submission_d111f5d016d0492ebc68c396c9b9bbab.csv", index=False)
print(f"Submission saved to ./submission/submission_d111f5d016d0492ebc68c396c9b9bbab.csv")
print(f"Submission shape: {submission_df.shape}")

# Final output
print(f"Final Validation Score: {final_val_logloss}")
