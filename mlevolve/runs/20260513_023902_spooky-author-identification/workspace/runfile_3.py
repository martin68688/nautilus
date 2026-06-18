import os
os.sched_setaffinity(0, {4, 6, 7, 8, 9, 11, 12, 13, 14, 15})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Encode labels
label_encoder = LabelEncoder()
train_df["author_encoded"] = label_encoder.fit_transform(train_df["author"])
num_classes = len(label_encoder.classes_)

# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].values,
    train_df["author_encoded"].values,
    test_size=0.15,
    random_state=42,
    stratify=train_df["author_encoded"],
)

test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Load tokenizer and model
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize
max_length = 512
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="pt",
)
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="pt",
)
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="pt",
)

# Create datasets and dataloaders
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    torch.tensor(val_labels, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"]
)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

# Training configuration
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixed precision
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

# Training loop with best model checkpointing
num_epochs = 20
best_val_loss = float("inf")
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_trues.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_trues = np.concatenate(val_trues)
    val_loss = log_loss(val_trues, val_preds)

    print(
        f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Val Log Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

# Load best model for final predictions
model.load_state_dict(best_model_state)
print(f"Loaded best model with validation log loss: {best_val_loss:.4f}")

# Test predictions
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.concatenate(test_preds)

# Create submission
submission = pd.DataFrame(test_preds, columns=label_encoder.classes_)
submission["id"] = test_ids
submission = submission[["id", "EAP", "HPL", "MWS"]]

# Ensure probabilities sum to 1
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums

# Clip to avoid extremes
epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col].clip(epsilon, 1 - epsilon)

os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission_c2d00fc7c5e04e059cd74117e2752213.csv", index=False)

# Final validation score
val_preds_final = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        val_preds_final.append(probs.cpu().numpy())

val_preds_final = np.concatenate(val_preds_final)
score = log_loss(val_trues, val_preds_final)
print(f"Final Validation Score: {score}")