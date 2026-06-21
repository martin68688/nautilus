import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Setup directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


def prepare_data():
    base_path = "./input/spooky-author-identification/prepared/public"
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))

    le = LabelEncoder()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in splitter.split(train_df, train_df["author"]):
        train_set = train_df.iloc[train_idx].copy()
        val_set = train_df.iloc[val_idx].copy()

    train_set["label"] = le.fit_transform(train_set["author"])
    val_set["label"] = le.transform(val_set["author"])

    train_set.to_csv("./working/train_processed.csv", index=False)
    val_set.to_csv("./working/val_processed.csv", index=False)
    test_df.to_csv("./working/test_processed.csv", index=False)
    np.save("./working/label_encoder_classes.npy", le.classes_)


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_pipeline():
    prepare_data()

    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=3)
    config.classifier_dropout = 0.3
    config.dropout = 0.3
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", config=config
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    train_set = pd.read_csv("./working/train_processed.csv")
    val_set = pd.read_csv("./working/val_processed.csv")
    test_df = pd.read_csv("./working/test_processed.csv")

    train_ds = SpookyDataset(
        train_set["text"].tolist(), train_set["label"].tolist(), tokenizer
    )
    val_ds = SpookyDataset(
        val_set["text"].tolist(), val_set["label"].tolist(), tokenizer
    )
    test_ds = SpookyDataset(test_df["text"].tolist(), None, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits

            # Manual label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            target = F.one_hot(labels, num_classes=3).float()
            smoothed_targets = (1 - 0.1) * target + 0.1 / 3
            loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_losses.append(model(**batch).loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1} Complete. Loss: {total_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "./working/best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("./working/best_model.pth"))
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            val_preds.append(F.softmax(model(**batch).logits, dim=-1).cpu().numpy())

    val_probs = np.vstack(val_preds)
    score = log_loss(val_set["label"].values, val_probs)

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            if "labels" in batch:
                batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            test_preds.append(F.softmax(model(**batch).logits, dim=-1).cpu().numpy())

    final_probs = np.vstack(test_preds)
    submission = pd.DataFrame(final_probs, columns=["EAP", "HPL", "MWS"])
    submission["id"] = test_df["id"]
    submission[["id", "EAP", "HPL", "MWS"]].to_csv(
        "./submission/submission.csv", index=False
    )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    run_pipeline()