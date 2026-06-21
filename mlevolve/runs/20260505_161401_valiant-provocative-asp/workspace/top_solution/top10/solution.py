import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Configuration
INPUT_DIR = "./input/spooky-author-identification/prepared/public"
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./submission", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data():
    train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
    test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["author"])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(train_df, train_df["label"]))
    train_set, val_set = train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[
        val_idx
    ].reset_index(drop=True)

    train_set.to_csv(f"{OUTPUT_DIR}/train_processed.csv", index=False)
    val_set.to_csv(f"{OUTPUT_DIR}/val_processed.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test_processed.csv", index=False)
    return train_set, val_set, test_df


class SpookyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df, self.tokenizer, self.max_len = df, tokenizer, max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if "label" in self.df.columns:
            item["labels"] = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return item


# Logic execution
train_df, val_df, test_df = preprocess_data()
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
).to(device)

train_loader = DataLoader(
    SpookyDataset(train_df, tokenizer), batch_size=8, shuffle=True
)
val_loader = DataLoader(SpookyDataset(val_df, tokenizer), batch_size=8, shuffle=False)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * 3
)

best_loss = float("inf")
patience = 0
for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += F.cross_entropy(outputs.logits, batch["labels"]).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Complete - Validation Loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))
model.eval()
test_loader = DataLoader(SpookyDataset(test_df, tokenizer), batch_size=8, shuffle=False)
preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        probs = F.softmax(model(**batch).logits, dim=1)
        preds.append(probs.cpu().numpy())

submission = pd.concat(
    [test_df[["id"]], pd.DataFrame(np.vstack(preds), columns=["EAP", "HPL", "MWS"])],
    axis=1,
)
submission.to_csv("./submission/submission.csv", index=False)
score = best_loss
print(f"Final Validation Score: {score}")