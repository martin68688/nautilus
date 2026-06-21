import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer parallelization issues
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import AdamW
from sklearn.metrics import log_loss
import numpy as np

# Step 1: Data Processing and Feature Engineering
train_df = pd.read_csv("./input/spooky-author-identification/prepared/public/train.csv")
test_df = pd.read_csv("./input/spooky-author-identification/prepared/public/test.csv")

model_id = "distilbert-base-uncased"  # Smaller model for faster training
tokenizer = AutoTokenizer.from_pretrained(model_id)


def preprocess_text(text):
    return text.replace("\n", " ").replace("\r", " ")


train_df["text"] = train_df["text"].apply(preprocess_text)
test_df["text"] = test_df["text"].apply(preprocess_text)

# Enable tensor cores for better GPU performance
torch.set_float32_matmul_precision('high')

train_encodings = tokenizer(
    train_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128,  # Reduced max length for efficiency
    return_tensors="pt",
)
test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128,  # Reduced max length for efficiency
    return_tensors="pt",
)

label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_labels = torch.tensor(train_df["author"].map(label_map).values)

# Include attention masks in train/val split
attention_masks = train_encodings['attention_mask']
X_train, X_val, masks_train, masks_val, y_train, y_val = train_test_split(
    train_encodings.input_ids,
    attention_masks,
    train_labels,
    test_size=0.2,
    random_state=42
)

# Step 2: Model Design
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 3: Training and Evaluation
# Update datasets to include attention masks
train_dataset = torch.utils.data.TensorDataset(X_train, masks_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, masks_val, y_val)
test_dataset = torch.utils.data.TensorDataset(test_encodings.input_ids, test_encodings['attention_mask'])

# Reduce batch size to fit within time constraints
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

best_loss = float("inf")
patience = 3
no_improve_count = 0

accumulation_steps = 2  # Allows effective batch size of 32 while using half memory
for epoch in range(3):  # Further reduced epochs to meet time constraints
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, masks, labels) in enumerate(train_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        outputs = model(inputs, attention_mask=masks).logits
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps

    model.eval()
    val_preds = []
    val_labels_np = y_val.numpy()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, masks, labels in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs, attention_mask=masks).logits
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            loss = criterion(outputs, labels.to(device))
            val_loss += loss.item()

    val_preds = np.array(val_preds)
    # Convert predictions to DataFrame for proper format
    val_preds_df = pd.DataFrame(val_preds, columns=["EAP", "HPL", "MWS"])
    # Convert labels to one-hot encoding
    val_labels_onehot = pd.get_dummies(np.array([["EAP", "HPL", "MWS"][x] for x in val_labels_np]))
    val_log_loss = log_loss(val_labels_onehot, val_preds_df)

    if val_log_loss < best_loss:
        best_loss = val_log_loss
        torch.save(model.state_dict(), "./working/best_model.pt")
        no_improve_count = 0
    else:
        no_improve_count += 1

    print(
        f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Log Loss {val_log_loss:.4f}"
    )

    if no_improve_count >= patience:
        print("Early stopping")
        break

model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()
test_preds = []
with torch.no_grad():
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs, attention_mask=masks).logits
        preds = torch.softmax(outputs, dim=1).cpu().numpy()
        test_preds.extend(preds)

test_ids = test_df["id"].values
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": [p[0] for p in test_preds],
        "HPL": [p[1] for p in test_preds],
        "MWS": [p[2] for p in test_preds],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {best_loss}")
